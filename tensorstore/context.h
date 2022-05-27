// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORSTORE_CONTEXT_H_
#define TENSORSTORE_CONTEXT_H_

/// \file Interfaces for creating and using context resources.

#include <assert.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context_impl_base.h"  // IWYU pragma: export
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Immutable collection of lazily-constructed "resources".
///
/// This is used as a holder for, among other things, shared thread pools, cache
/// pools, and authentication credentials used by various components in
/// TensorStore.
///
/// A resource is an arbitrary C++ object that may be accessed using a
/// `Context::Resource` handle, which provides a const, reference counted view
/// of the object.  Note that while the object itself is const, it may refer to
/// mutable state, such as a cache.  Resources are constructed lazily within the
/// `Context` object from an unresolved `Resource` spec (which is created from a
/// JSON representation) when first accessed.
///
/// Context resources are used with the TensorStore library to specify
/// TensorStore driver and kvstore driver configuration options that do not
/// affect the identity of the KvStore or TensorStore; in general, the contents
/// of a KvStore or TensorStore should be identical for any valid context.
/// For example, a file or bucket path for accessing data should not be
/// specified as a context resource, while authentication credentials, caching,
/// and concurrency options should be specified as context resources.
///
/// Example usage::
///
///     auto spec_result = Context::Spec::FromJson({
///       {"data_copy_concurrency", {{"limit", 4}}},
///       {"data_copy_concurrency#a", {{"limit", 8}}},
///     });
///     if (!spec_result) {
///       // Handle error
///     }
///     auto resource_spec_result =
///         Context::Resource<internal::DataCopyConcurrencyResource>
///         ::FromJson("data_copy_concurrency");
///     if (!resource_spec_result) {
///       // Handle error
///     }
///     auto context = Context(*spec_result);
///     auto resource_result = context.GetResource(
///         resource_spec_result->value());
///     if (!resource_result) {
///       // Handle error
///     }
///     // Can access executor member of context resource.
///     (*resource_result)->executor([] {
///       // task body
///     });
///     // Access other executor.
///     auto resource2 = context.GetResource(
///         Context::Resource<internal::DataCopyConcurrencyResource>
///         ::FromJson("data_copy_concurrency#a").value()).value();
///     resource2->executor([] { /* task body */ });
///
/// `Context::Spec` and unresolved `Context::Resource` instances serve as
/// intermediate representations between JSON and the actual context and context
/// resource, respectively.  This indirection makes it possible to validate a
/// context or resource specification without actually constructing the
/// resource; this is especially useful if one process/machine prepares a
/// specification for use by another process/machine.  It also makes it possible
/// to properly handle shared resources when serializing back to JSON.
///
/// The `Context::Resource` type is actually a variant type that can hold either
/// a context resource or a context resource spec.
/// ``Context::Resource<Provider>::FromJson`` returns an unresolved context
/// resource spec, which specifies any parameters of the context resource
/// (e.g. memory limits for a cache pool) but does not represent a specific
/// resource (e.g. the actual cache pool).  *Binding* the context resource spec
/// with a `Context` converts the context resource spec to an actual context
/// resource.  *Unbinding* a context resource converts it back to a context
/// resource spec.  These binding and unbinding operations can also be applied
/// to data structures like `tensorstore::Spec` and `tensorstore::kvstore::Spec`
/// that indirectly hold context resources.
///
/// \ingroup core
class Context {
 public:
  /// JSON serialization options.
  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  /// Parsed specification of a collection of context resources.
  class Spec {
   public:
    /// Constructs an empty context spec.
    Spec() = default;

    TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, FromJsonOptions,
                                            ToJsonOptions)

   private:
    friend class internal_context::Access;
    friend class Context;
    friend class internal::ContextSpecBuilder;
    internal_context::ContextSpecImplPtr impl_;
  };

  /// Variant type that either specifies an unresolved context resource, or
  /// holds a shared handle to a context resource.
  ///
  /// Unresolved context resources may either be specified directly or by a
  /// named reference, and are resolved by calling `Context::GetResource`.
  template <typename Provider>
  class Resource {
    using ResourceImpl = internal_context::ResourceImpl<Provider>;

   public:
    /// JSON serialization options.
    using ToJsonOptions = JsonSerializationOptions;
    using FromJsonOptions = JsonSerializationOptions;

    /// Constructs an invalid handle.
    Resource() = default;

    /// Returns a resource spec that refers to the default value within the
    /// `Context`.
    ///
    /// The returned spec is guaranteed to be valid.
    static Resource<Provider> DefaultSpec() {
      Resource<Provider> r;
      r.impl_ = internal_context::DefaultResourceSpec(Provider::id);
      return r;
    }

    /// Returns a pointer to the resource object of type ``Provider::Resource``,
    /// or `nullptr` if no resource is bound.
    auto* get() const noexcept {
      return has_resource()
                 ? &(static_cast<ResourceImpl*>(impl_.get().get())->value_)
                 : nullptr;
    }
    auto* operator->() const noexcept { return get(); }
    auto& operator*() const noexcept {
      assert(has_resource());
      return *get();
    }

    /// Returns `true` if this is not null and corresponds to a bound resource
    /// (rather than a resource spec).
    bool has_resource() const { return impl_.get() && impl_.get().tag() == 0; }

    /// Returns `true` if this corresponds to a resource spec or bound resource.
    bool valid() const { return !!impl_; }

    /// Checks if `a` and `b` refer to the same resource or resource spec.
    friend bool operator==(const Resource& a, const Resource& b) {
      return a.impl_ == b.impl_;
    }
    friend bool operator!=(const Resource& a, const Resource& b) {
      return !(a == b);
    }

    /// Converts to a JSON representation.
    Result<::nlohmann::json> ToJson(
        const ToJsonOptions& options = ToJsonOptions{}) const {
      return internal_json_binding::ToJson(
          static_cast<const Resource&>(*this),
          internal_json_binding::DefaultBinder<>, options);
    }

    /// Returns a resource spec from a JSON representation.
    static Result<Resource<Provider>> FromJson(
        ::nlohmann::json j,
        const FromJsonOptions& options = FromJsonOptions{}) {
      return internal_json_binding::FromJson<Resource<Provider>>(
          std::move(j), internal_json_binding::DefaultBinder<>, options);
    }

    // Support for json binding via internal_json_binding::DefaultBinder<>.
    static constexpr internal_context::ResourceJsonBinderImpl<Provider,
                                                              Resource>
        default_json_binder = {};

    /// If `!this->has_resource()`, resolves this to a resource using the
    /// specified context.  Otherwise, does nothing.
    ///
    /// Upon successful return, `has_resource() == true`.
    absl::Status BindContext(const Context& context) {
      TENSORSTORE_ASSIGN_OR_RETURN(*this, context.GetResource(*this));
      return absl::OkStatus();
    }

    // If `!this->has_resource()`, resolves this to a resource using the
    // specified context.  Otherwise, does nothing.
    //
    // Upon successful return, `has_resource() == true`.
    absl::Status BindContext(internal::ContextResourceCreationContext context);

    // If `this->has_resource()`, converts this to a resource spec.  Otherwise,
    // does nothing.
    //
    // \post `has_resource() == false`
    void UnbindContext(
        const internal::ContextSpecBuilder& context_spec_builder);

    /// If this is non-null, resets it to `DefaultSpec()`.  If this is null, it
    /// remains null.
    ///
    /// \post `has_resource() == false`
    void StripContext() { internal_context::StripContext(impl_); }

   private:
    friend class Context;
    friend class internal::ContextSpecBuilder;
    friend class internal_context::Access;
    friend struct internal_context::ResourceJsonBinderImpl<Provider, Resource>;

    internal_context::ResourceOrSpecPtr impl_;
  };

  /// Constructs a null context.
  ///
  /// \id null
  Context() = default;

  /// Constructs a new default context.
  static Context Default();

  /// Constructs a context from a `Spec`.
  ///
  /// The actual resources are created lazily as needed.
  ///
  /// \param spec The context spec.
  /// \param parent The parent context to extend.  Specifying a null context is
  ///     equivalent to specifying `Default()`.
  /// \id spec, parent
  explicit Context(const Spec& spec, Context parent = {});

  /// Constructs a context from a JSON spec.
  ///
  /// \param json_spec The JSON spec.
  /// \param parent The parent context to extend.  Specifying a null context is
  ///     equivalent to specifying `Default()`.
  /// \param options Options for parsing `json_spec`.
  static Result<Context> FromJson(::nlohmann::json json_spec,
                                  Context parent = {},
                                  FromJsonOptions options = {});

  /// Returns a resource or the default resource for a given provider.
  ///
  /// If the nullary overload is called, returns the default resource for
  /// `Provider`.
  ///
  /// \tparam Provider Context resource provider type.  Must be specified when
  ///     not using the `resource_spec` overload.
  /// \param resource_spec The resource spec.  May be null, in which case a null
  ///     resource is returned.
  /// \param json_spec The JSON resource spec.
  /// \dchecks This context is not null.
  template <typename Provider>
  Result<Resource<Provider>> GetResource(
      const Resource<Provider>& resource_spec) const {
    Resource<Provider> resource;
    TENSORSTORE_RETURN_IF_ERROR(internal_context::GetOrCreateResource(
        impl_.get(), resource_spec.impl_.get(), /*trigger=*/nullptr,
        /*resource=*/resource.impl_));
    return resource;
  }
  template <typename Provider>
  Result<Resource<Provider>> GetResource(
      const ::nlohmann::json& json_spec) const {
    TENSORSTORE_ASSIGN_OR_RETURN(auto spec,
                                 Resource<Provider>::FromJson(json_spec));
    return GetResource(spec);
  }
  template <typename Provider>
  Result<Resource<Provider>> GetResource() {
    return GetResource<Provider>(Provider::id);
  }

  /// Returns `true` if this is not a null context.
  ///
  /// \id bool
  explicit operator bool() const { return static_cast<bool>(impl_); }

  /// Returns `true` if the two context objects refer to the same underlying
  /// shared state.
  friend bool operator==(const Context& a, const Context& b) {
    return a.impl_ == b.impl_;
  }
  friend bool operator!=(const Context& a, const Context& b) {
    return !(a == b);
  }

  /// Returns the `Spec` used to create this context.
  Context::Spec spec() const;

  /// Returns the parent `Context`, if any.
  Context parent() const;

 private:
  friend class internal_context::Access;
  internal_context::ContextImplPtr impl_;
};

/// Specifies how context bindings should be handled for Spec-like types, such
/// as `tensorstore::kvstore::Spec` and `tensorstore::Spec`.
///
/// This is an "option" type that can be used with `TensorStore::spec`,
/// `KvStore::spec`, and related methods.
///
/// \relates Context
enum class ContextBindingMode : unsigned char {
  /// Context binding mode is unspecified.
  unspecified,

  /// Retain all bound context resources and unbound context resource specs.
  retain,

  /// Any bound context resources are converted to context resource specs that
  /// fully capture the graph of shared context resources and interdependencies.
  /// Re-binding/re-opening the resultant spec will result in a new graph of new
  /// context resources that is isomorphic to the original graph of context
  /// resources.  The resultant spec will not refer to any external context
  /// resources; consequently, binding it to any specific context will have the
  /// same effect as binding it to a default context.
  unbind,

  /// Any bound context resources and unbound context resource specs are
  /// replaced by default context resource specs.  If the resultant spec is
  /// re-opened with/re-bound to a new context, it will use the default context
  /// resources specified by that context.
  strip,
};

/// Indicates the binding state of context resources within a Spec-like type,
/// such as `tensorstore::Spec` and `tensorstore::kvstore::Spec`.
///
/// \relates Context
enum class ContextBindingState : unsigned char {
  /// All resources are unbound.
  unbound,
  /// Binding state is unknown, some resources may be bound and some may be
  /// unbound.
  unknown,
  /// All resources are bound.
  bound
};

/// \relates ContextBindingMode
constexpr ContextBindingMode retain_context = ContextBindingMode::retain;
constexpr ContextBindingMode unbind_context = ContextBindingMode::unbind;
constexpr ContextBindingMode strip_context = ContextBindingMode::strip;

namespace internal {

/// Used to create a `ContextSpec` from an existing collection of
/// `Context::Resource` objects.
///
/// \threadsafety Thread compatible
class ContextSpecBuilder {
 public:
  /// Constructs a null builder.
  ContextSpecBuilder() = default;

  /// Returns `true` if this is not a null builder.
  explicit operator bool() const { return static_cast<bool>(impl_); }

  /// Constructs a new `ContextSpecBuilder`.
  ///
  /// If `parent` is a null builder, returns a new root builder.  Otherwise,
  /// returns a new child builder of the specified `parent`.
  ///
  /// Currently, all shared resources are defined in the `spec()` of the root
  /// builder.  Calling `spec()` on a non-root builder always results in an
  /// empty `Context::Spec`.
  ///
  /// Any resources specified by `existing_spec` are also included in the
  /// returned `spec`.  Any additional shared resources will be assigned
  /// identifiers distinct from any in `existing_spec`.
  static ContextSpecBuilder Make(ContextSpecBuilder parent = {},
                                 Context::Spec existing_spec = {});

  /// Registers a resource with this builder, and returns a `ResourceSpec` that
  /// refers to it.
  ///
  /// If `resource` is shared (meaning this method is called multiple times
  /// within the same `ContextSpecBuilder` tree with the same `resource`) or if
  /// `resource` was constructed from a named context resource spec, the
  /// returned `Resource` will be a named reference to the actual resource
  /// specification defined in the `Context::Spec` returned by `spec()`.
  ///
  /// Otherwise, the returned `Resource` will specify the resource directly.
  ///
  /// The returned `Resource` must not be used until all `ContextSpecBuilder`
  /// objects within the tree have been destroyed.
  template <typename Provider>
  Context::Resource<Provider> AddResource(
      const Context::Resource<Provider>& resource) const {
    Context::Resource<Provider> resource_spec;
    resource_spec.impl_ =
        internal_context::AddResourceOrSpec(*this, resource.impl_.get());
    return resource_spec;
  }

  /// Returns the `Context::Spec` that defines the shared resources registered
  /// with this builder or a descendant builder.  Also includes all resources
  /// specified by the `existing_spec` passed to `Make`.
  Context::Spec spec() const;

 private:
  friend class internal_context::Access;
  internal_context::BuilderImplPtr impl_;
  internal_context::ContextSpecImplPtr spec_impl_;
};

TENSORSTORE_DECLARE_JSON_BINDER(ContextSpecDefaultableJsonBinder, Context::Spec,
                                JsonSerializationOptions,
                                JsonSerializationOptions)

/// Indicates whether `context` is to be used for binding only context resource
/// specs explicitly marked for immediate binding, rather than all context
/// resources.  Context resource specs not marked for immediate binding
/// (i.e. not instances of `BuilderResourceSpec`) should remain unbound.
bool IsPartialBindingContext(const Context& context);

/// Indicates whether `ContextSpecBuilder::AddResource` should return context
/// resource specs tagged to indicate whether the input was a spec or resource.
///
/// This is used when serializing Spec-like types to record which resources were
/// bound.
inline bool GetRecordBindingState(const internal::ContextSpecBuilder& builder) {
  return internal_context::Access::impl(builder).get().tag() != 0;
}

/// Sets the value returned by `GetRecordBindingState`.
///
/// The value is also inherited by any child builders.
void SetRecordBindingState(internal::ContextSpecBuilder& builder,
                           bool record_binding_state);

/// Binds context resources with a copy-on-write pointer type that has a nested
/// `context_spec_` and `context_binding_state_`.
///
/// This is used for `DriverSpecPtr` and `kvstore::DriverSpecPtr`.
///
/// \param ptr Object to bind, typically a `DriverSpecPtr` or
///     `kvstore::DriverSpecPtr`.
/// \param context The context to bind.
/// \requires `ptr->Clone()` returns a copy of the object of type `Ptr`
/// \requires `ptr->use_count()` returns the reference count.
/// \requires `ptr->BindContext(const Context&)` method.
template <typename Ptr>
absl::Status BindContextCopyOnWriteWithNestedContext(Ptr& ptr,
                                                     const Context& context) {
  if (!ptr) return absl::OkStatus();
  using internal_context::Access;
  {
    auto& orig_obj = *ptr;
    if (Access::context_binding_state(orig_obj) == ContextBindingState::bound) {
      return absl::OkStatus();
    }
    if (orig_obj.use_count() != 1) ptr = orig_obj.Clone();
  }
  using T = internal::remove_cvref_t<decltype(*ptr)>;
  auto& obj = const_cast<T&>(*ptr);
  Access::context_binding_state(obj) = ContextBindingState::unknown;

  // Binds context resources via Ptr::BindContext
  // Takes into account `IsPartialBindingContext(context)`
  if (context && IsPartialBindingContext(context)) {
    // Partial binding; avoid constructing a child context.
    TENSORSTORE_RETURN_IF_ERROR(obj.BindContext(context));
  } else {
    // Full binding uses a child context.
    Context child_context(Access::context_spec(obj),
                          context ? context : Context::Default());
    TENSORSTORE_RETURN_IF_ERROR(obj.BindContext(child_context));
    Access::context_spec(obj) = {};
    Access::context_binding_state(obj) = ContextBindingState::bound;
  }
  return absl::OkStatus();
}

/// Inverse of `BindContextCopyOnWriteWithNestedContext`.
///
/// Unbinds resources by calling ptr->UnbindContext(const
/// internal::ContextSpecBuilder&)
///
/// \param ptr  Object to unbind, typically a `DriverSpecPtr` or
///      `kvstore::DriverSpecPtr`.
/// \param context_builder The ContextBuilder used for unbinding.
/// \requires `ptr->Clone()` returns a copy of the object of type `Ptr`
/// \requires `ptr->use_count()` returns the reference count.
/// \requires `ptr->UnbindContext(const ContextSpecBuilder&)` method.
template <typename Ptr>
void UnbindContextCopyOnWriteWithNestedContext(
    Ptr& ptr, const ContextSpecBuilder& context_builder) {
  if (!ptr) return;
  using internal_context::Access;
  {
    auto& orig_obj = *ptr;
    if (Access::context_binding_state(orig_obj) ==
        ContextBindingState::unbound) {
      return;
    }
    if (orig_obj.use_count() != 1) ptr = orig_obj.Clone();
  }
  using T = internal::remove_cvref_t<decltype(*ptr)>;
  auto& obj = const_cast<T&>(*ptr);
  // Unbinds context resources via Ptr::UnbindContext
  auto child_builder = internal::ContextSpecBuilder::Make(
      context_builder, std::move(Access::context_spec(obj)));
  Access::context_spec(obj) = child_builder.spec();
  obj.UnbindContext(
      const_cast<const internal::ContextSpecBuilder&>(child_builder));
  Access::context_binding_state(obj) = ContextBindingState::unbound;
}

/// Strips context resources from a copy-on-write pointer type that has a nested
/// `context_spec_` and `context_binding_state_`.
///
/// This is used for `DriverSpecPtr` and `kvstore::DriverSpecPtr`.
template <typename Ptr>
void StripContextCopyOnWriteWithNestedContext(Ptr& ptr) {
  if (!ptr) return;
  using internal_context::Access;
  {
    auto& orig_obj = *ptr;
    if (orig_obj.use_count() != 1) ptr = orig_obj.Clone();
  }
  using T = internal::remove_cvref_t<decltype(*ptr)>;
  auto& obj = const_cast<T&>(*ptr);
  Access::context_spec(obj) = {};
  obj.StripContext();
  Access::context_binding_state(obj) = ContextBindingState::unbound;
}

/// Applies a context binding mode operation to a type (like
/// `kvstore::DriverSpecPtr` or `Spec`) that supports `UnbindContext` and
/// `StripContext`.
///
/// \param ptr Object to which the operation should be applied.
/// \param mode Binding mode that was requested.
/// \param default_mode Mode to use if
///     `mode == ContextBindingMode::unspecified`.
template <typename Ptr>
void ApplyContextBindingMode(Ptr& ptr, ContextBindingMode mode,
                             ContextBindingMode default_mode) {
  if (mode == ContextBindingMode::unspecified) mode = default_mode;
  switch (mode) {
    case ContextBindingMode::unbind:
      ptr.UnbindContext();
      break;
    case ContextBindingMode::strip:
      ptr.StripContext();
      break;
    case ContextBindingMode::retain:
    case ContextBindingMode::unspecified:
      break;
  }
}

/// Context object for use by `Create` methods of context resource traits
/// classes.
class ContextResourceCreationContext {
 public:
  // The following members are internal use only.

  internal_context::ContextImpl* context_ = nullptr;
  internal_context::ResourceContainer* trigger_ = nullptr;
};

}  // namespace internal

namespace internal_json_binding {

/// JSON binder for types (like `Spec` and `kvstore::Spec`) that contain context
/// resources and/or context resource specs and support a
/// `context_binding_state()` method.
template <typename Binder>
auto NestedContextJsonBinder(Binder binder) {
  return [binder = std::move(binder)](auto is_loading,
                                      const JsonSerializationOptions& options,
                                      auto* obj, auto* j) {
    if constexpr (!is_loading) {
      if (obj->context_binding_state() != ContextBindingState::unbound) {
        auto copy = *obj;
        internal::ContextSpecBuilder spec_builder;
        if (options.preserve_bound_context_resources_) {
          internal::SetRecordBindingState(spec_builder, true);
        }
        copy.UnbindContext(spec_builder);
        return binder(is_loading, options, &copy, j);
      }
    }
    return binder(is_loading, options, obj, j);
  };
}

}  // namespace internal_json_binding

template <typename Provider>
absl::Status Context::Resource<Provider>::BindContext(
    internal::ContextResourceCreationContext context) {
  return internal_context::GetOrCreateResource(context.context_, impl_.get(),
                                               context.trigger_, impl_);
}

template <typename Provider>
void Context::Resource<Provider>::UnbindContext(
    const internal::ContextSpecBuilder& context_spec_builder) {
  *this = context_spec_builder.AddResource(*this);
}

namespace serialization {

template <typename Provider>
struct Serializer<Context::Resource<Provider>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink,
                                   const Context::Resource<Provider>& value) {
    return internal_context::EncodeContextResourceOrSpec(
        sink, internal_context::Access::impl(value));
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   Context::Resource<Provider>& value) {
    return internal_context::DecodeContextResourceOrSpec(
        source, Provider::id, internal_context::Access::impl(value));
  }
};

}  // namespace serialization

namespace garbage_collection {
template <typename Provider>
struct GarbageCollection<Context::Resource<Provider>> {
  constexpr static bool required() { return false; }
};
}  // namespace garbage_collection

namespace internal {
template <typename Provider>
struct CacheKeyEncoder<Context::Resource<Provider>> {
  static void Encode(std::string* out,
                     const Context::Resource<Provider>& value) {
    internal::EncodeCacheKey(out,
                             reinterpret_cast<std::uintptr_t>(value.get()));
  }
};
}  // namespace internal

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::Context::Spec)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::Context)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::Context::Spec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::Context)

namespace std {
// Specialization of `std::pointer_traits` for `Context::Resource`.
//
// Note that we can't define a nested `element_type` within
// `Context::Resource`, which would eliminate the need for this specialization,
// because we need to allow `Context::Resource` to be instantiated even if
// `Provider` is incomplete (to allow recursive context references, for
// example).
template <typename Provider>
struct pointer_traits<tensorstore::Context::Resource<Provider>> {
  using pointer = tensorstore::Context::Resource<Provider>;
  using element_type = typename Provider::Resource;
  using difference_type = std::ptrdiff_t;
};
}  // namespace std

#endif  // TENSORSTORE_CONTEXT_H_
