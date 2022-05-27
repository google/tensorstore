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

#ifndef TENSORSTORE_CONTEXT_IMPL_BASE_H_
#define TENSORSTORE_CONTEXT_IMPL_BASE_H_

/// \file
///
/// Context implementation details and forward declarations required by
/// `context.h`.

// IWYU pragma: private, include "third_party/tensorstore/context.h"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
class Context;

namespace internal {
class ContextResourceCreationContext;
class ContextSpecBuilder;
}  // namespace internal
namespace internal_context {

/// Provides access to implementation details of `Context`.  For internal use
/// only.
class Access {
 public:
  template <typename T>
  static auto impl(T&& x) -> decltype(auto) {
    // Note: return expression is parenthesized to ensure `decltype(auto)`
    // resolves to a reference type.
    return (static_cast<T&&>(x).impl_);
  }

  template <typename T>
  static auto context_spec(T&& x) -> decltype(auto) {
    // Note: return expression is parenthesized to ensure `decltype(auto)`
    // resolves to a reference type.
    return (static_cast<T&&>(x).context_spec_);
  }

  template <typename T>
  static auto context_binding_state(T&& x) -> decltype(auto) {
    // Note: return expression is parenthesized to ensure `decltype(auto)`
    // resolves to a reference type.
    return (static_cast<T&&>(x).context_binding_state_);
  }
};

/// Immutable representation of `Context::Spec` held via reference-counted
/// pointer.  Holds a hash table of pointers to `ResourceSpecImplBase`
/// objects that were obtained from a context JSON specification.
class ContextSpecImpl;

class ResourceOrSpecBase;
class ResourceImplBase;
class ResourceSpecImplBase;

/// Representation of `Context` held via reference-counted pointer.  Logically
/// immutable.  References a `ContextSpecImpl` and holds a hash table of
/// pointers to `ResourceContainer` objects, which represent
/// lazily-constructed resources corresponding to `ContextImpl`
class ContextImpl;

/// Manages the lazy construction of a context resource.
class ResourceContainer;

/// Representation of a `ContextSpecBuilder` held via reference-counted pointer.
class BuilderImpl;

void intrusive_ptr_increment(ContextSpecImpl* p);
void intrusive_ptr_decrement(ContextSpecImpl* p);

void intrusive_ptr_increment(ContextImpl* p);
void intrusive_ptr_decrement(ContextImpl* p);

/// Tagged pointer to either a `ResourceImplBase` or
/// `ResourceSpecImplBase` (or to `nullptr`).
///
/// The tag bit 0 indicates the type.
///
/// The tag bit 1 indicates a resource spec that should be re-bound immediately
/// upon deserialization.  This is used in conjunction with
/// `JsonSerializationOptions::preserve_bound_context_resources_` and
/// `BuilderImpl::record_binding_state_`.  Pointers with tag bit 1 set are
/// never exposed publicly; they are only used internally during serialization
/// and deserialization.
using ResourceOrSpecTaggedPtr = internal::TaggedPtr<ResourceOrSpecBase, 2>;

/// Returns `true` if `ptr` points to a `ResourceImplBase`, according to
/// the tag.
///
/// This may return `true` even if `ptr` is `nullptr`.
inline bool IsResource(ResourceOrSpecTaggedPtr ptr) {
  return ptr.tag<0>() == false;
}

inline bool IsImmediateBindingResourceSpec(ResourceOrSpecTaggedPtr ptr) {
  return ptr.tag<1>();
}

/// IntrusivePtr traits used by `ResourceOrSpecPtr`.
struct ResourceOrSpecPtrTraits {
  template <typename>
  using pointer = ResourceOrSpecTaggedPtr;
  static void increment(ResourceOrSpecBase* p);
  static void decrement(ResourceOrSpecBase* p);
};

/// IntrusivePtr traits used by `ResourceImplWeakPtr`.
struct ResourceImplWeakPtrTraits {
  template <typename>
  using pointer = ResourceImplBase*;
  static void increment(ResourceOrSpecBase* p);
  static void decrement(ResourceOrSpecBase* p);
};

/// IntrusivePtr traits used by `ResourceImplStrongPtr`.
struct ResourceImplStrongPtrTraits {
  template <typename>
  using pointer = ResourceImplBase*;
  static void increment(ResourceImplBase* p);
  static void decrement(ResourceImplBase* p);
};

/// Used by `internal::ContextSpecBuilder`.  The tag bit is set to 1 if the
/// context binding state of resources should be preserved, i.e. context
/// resource specs returned by `AddResource` that are derived from a
/// previously-bound resource should have the tag bit 1 set ( refer to
/// `ResourceOrSpecTaggedPtr` for details).
using BuilderImplTaggedPtr = internal::TaggedPtr<BuilderImpl, 1>;

/// IntrusivePtr traits used by `BuilderImplPtr`.
struct BuilderImplPtrTraits {
  template <typename>
  using pointer = BuilderImplTaggedPtr;
  static void increment(BuilderImplTaggedPtr p);
  static void decrement(BuilderImplTaggedPtr p);
};

using BuilderImplPtr =
    internal::IntrusivePtr<BuilderImpl, BuilderImplPtrTraits>;
using ContextSpecImplPtr = internal::IntrusivePtr<ContextSpecImpl>;
using ContextImplPtr = internal::IntrusivePtr<ContextImpl>;

/// Reference-counted smart pointer used internally to refer to a
/// `ResourceSpecImplBase`.
using ResourceSpecImplPtr = internal::IntrusivePtr<ResourceSpecImplBase>;

/// Reference-counted smart pointer used by `Context::Resource` to refer to
/// either a `ResourceSpecImplBase` or a `ResourceImplBase`.  When
/// pointing to a `ResourceImplBase`, this is a "weak pointer", like
/// `ResourceImplWeakPtr`.
using ResourceOrSpecPtr =
    internal::IntrusivePtr<ResourceOrSpecBase, ResourceOrSpecPtrTraits>;

/// Reference-counted smart pointer used internally to refer to a
/// `ResourceImplBase`.  This is a weak pointer: the context resource
/// provider may not preserve certain cached data when there are no strong
/// pointers remaining.
using ResourceImplWeakPtr =
    internal::IntrusivePtr<ResourceImplBase, ResourceImplWeakPtrTraits>;

/// Reference-counted smart pointer used (indirectly) by `ContextImpl` to refer
/// to a `ResourceImplBase`.  This is a strong pointer.
using ResourceImplStrongPtr =
    internal::IntrusivePtr<ResourceImplBase, ResourceImplStrongPtrTraits>;

/// Abstract base class for representing a registered context resource provider.
///
/// For a given `Traits` type, `ResourceProviderImpl<Traits>` is the
/// derived class that implements this interface.
class ResourceProviderImplBase {
 public:
  /// Unique provider id that identifies this provider in the JSON
  /// representation, e.g. "cache_pool".
  std::string_view id_;

  /// Returns a non-null pointer to a new spec that may be used to construct a
  /// resource for this provider in its default state.
  virtual ResourceSpecImplPtr Default() const = 0;

  /// Obtains a new spec from its JSON representation.
  ///
  /// \returns Non-null pointer to spec.
  virtual Result<ResourceSpecImplPtr> FromJson(
      const ::nlohmann::json& j, JsonSerializationOptions options) const = 0;

  /// Called when the first strong reference is acquired.  For example, the
  /// implementation may choose to retain additional caches in response.
  ///
  /// \param resource A resource constructed from this provider.
  virtual void AcquireContextReference(ResourceImplBase& resource) const = 0;

  /// Called when the last strong reference is released.  For example, the
  /// implementation may choose to release certain unused caches in response.
  ///
  /// \param resource A resource constructed from this provider.
  virtual void ReleaseContextReference(ResourceImplBase& resource) const = 0;

  virtual ~ResourceProviderImplBase();

 private:
  friend class ResourceImplBase;

  /// Returns a new spec corresponding to a resource.
  ///
  /// It is not necessary to set the `provider_`, `is_default_`, or `key_`
  /// fields of the returned spec, as they will be set automatically by
  /// `ResourceImplBase::UnbindContext`, which is the only caller of this
  /// method.
  ///
  /// \param resource A resource constructed from this provider.
  /// \param spec_builder Must be used to convert any nested context resources.
  /// \returns Non-null pointer to spec.
  virtual ResourceSpecImplPtr DoGetSpec(
      const ResourceImplBase& resource,
      const internal::ContextSpecBuilder& spec_builder) const = 0;
};

/// Reference-counted base class for either a context resource spec
/// (`ResourceSpecImplBase`) or a context resource
/// (`ResourceImplBase`).
///
/// Objects of this type may be referenced by a `ResourceOrSpecPtr`, where the
/// tag bit indicates whether the pointee is a context resource (0) or context
/// resource spec (1).
class ResourceOrSpecBase
    : public internal::AtomicReferenceCount<ResourceOrSpecBase> {
 public:
  virtual ~ResourceOrSpecBase();
};

/// Representation of a `Context::ResourceSpec` held via reference-counted
/// pointer.  Also used to represent a resource specification contained in a
/// `Context::Spec`.
///
/// For each resource provider type, the template machinery in the derived class
/// `ResourceProviderImpl<Traits>::SpecImpl` is the derived class that
/// holds the actual `Traits::Spec` object.
class ResourceSpecImplBase : public ResourceOrSpecBase {
 public:
  /// Creates a new context resource from this spec.
  ///
  /// \param creation_context May be used to resolve any nested context
  ///     resources.
  /// \returns Non-null pointer to resource.
  virtual Result<ResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context) = 0;

  /// Converts this spec to its JSON representation.
  virtual Result<::nlohmann::json> ToJson(JsonSerializationOptions options) = 0;

  /// Returns a copy of this resource spec, where all embedded context resources
  /// are also unbound.
  ///
  /// This ensures that any resources and resource identifiers are properly
  /// registered with `spec_builder`.
  ///
  /// The returned spec must *not* be an instance of `BuilderResourceSpec`, as
  /// that is used during serialization to indicate a resource that should be
  /// re-bound immediately upon deserialization.
  ///
  /// \returns Non-null pointer to spec.
  virtual ResourceSpecImplPtr UnbindContext(
      const internal::ContextSpecBuilder& spec_builder) = 0;

  virtual ~ResourceSpecImplBase();

  /// Specifies the key associated with this resource spec.  Must be empty if,
  /// and only if, the resource spec is specified inline outside a context spec.
  /// If `key_` is empty, no `Context::Spec` objects may point to this resource
  /// spec.
  std::string key_;

  /// Non-null pointer to provider associated with this spec.
  const ResourceProviderImplBase* provider_;

  bool is_default_ = false;
};

/// Representation of `Context::Resource` held via reference-counted pointer.
class ResourceImplBase : public ResourceOrSpecBase {
 public:
  ResourceImplBase(ResourceSpecImplPtr spec = {}) : spec_(std::move(spec)) {}
  virtual ~ResourceImplBase();

  /// Returns a context resource spec that may be used to recreate this
  /// resource.
  ///
  /// \param spec_builder Must be used to obtain the spec for any nested context
  ///     resources.
  /// \returns Non-null pointer to spec.
  ResourceSpecImplPtr UnbindContext(
      const internal::ContextSpecBuilder& spec_builder);

  /// Non-null pointer to spec from which this resource was constructed.
  ResourceSpecImplPtr spec_;
  // Protects access to `weak_creator_`.
  absl::Mutex mutex_;
  // Weak pointer to `ContextImpl` that created this resource.  Note that this
  // is a weak pointer, and if the `ContextImpl` is destroyed before this
  // resource is destroyed, `weak_context_` will be set to `nullptr`.  This is
  // used by tensorstore.distributed when serializing context objects to ensure
  // resource identity is consistent when a Context object is "shared" between
  // the controller and a worker.
  ContextImpl* weak_creator_ = nullptr;
};

/// For each resource provider type, `ResourceImpl<Provider>` is the
/// derived class of `ResourceImplBase` that holds the actual
/// `Provider::Resource` object.
///
/// This is parameterized by the `Provider` type rather than the `Traits` type
/// because it is needed by `Context::Resource`, which is also parameterized by
/// the `Provider` type rather than the `Traits` type.  The `Traits` type for a
/// given `Provider` type may only be defined in a separate source file that
/// actually registers the provider; code that merely uses the resource provider
/// does not need it.
template <typename Provider>
class ResourceImpl : public ResourceImplBase {
 public:
  using Resource = typename Provider::Resource;
  template <typename... Arg>
  ResourceImpl(ResourceSpecImplPtr spec, Arg&&... arg)
      : ResourceImplBase(std::move(spec)), value_(std::forward<Arg>(arg)...) {}

  Resource value_;
};

/// Implements the `ResourceProviderImplBase` type for a given context
/// resource provider traits type.
template <typename Traits>
class ResourceProviderImpl : public ResourceProviderImplBase {
 public:
  using Provider = typename Traits::Provider;
  template <typename... U>
  ResourceProviderImpl(U&&... arg) : traits_(std::forward<U>(arg)...) {
    id_ = Provider::id;
  }
  using Spec = typename Traits::Spec;
  using Resource = typename Provider::Resource;
  using ResourceImpl = internal_context::ResourceImpl<Provider>;
  class SpecImpl : public ResourceSpecImplBase {
   public:
    template <typename... U>
    SpecImpl(U&&... arg) : value_(std::forward<U>(arg)...) {}

    Result<ResourceImplStrongPtr> CreateResource(
        const internal::ContextResourceCreationContext& creation_context)
        override {
      auto result =
          static_cast<const ResourceProviderImpl*>(provider_)->traits_.Create(
              value_, creation_context);
      if (!result) {
        return std::move(result).status();
      }
      return ResourceImplStrongPtr(
          new ResourceImpl(ResourceSpecImplPtr(this), std::move(*result)));
    }

    Result<::nlohmann::json> ToJson(JsonSerializationOptions options) override {
      return internal_json_binding::ToJson(
          value_,
          static_cast<const ResourceProviderImpl*>(provider_)
              ->traits_.JsonBinder(),
          options);
    }

    ResourceSpecImplPtr UnbindContext(
        const internal::ContextSpecBuilder& spec_builder) final {
      ResourceSpecImplPtr new_spec_impl(new SpecImpl(value_));
      static_cast<const ResourceProviderImpl*>(provider_)
          ->traits_.UnbindContext(static_cast<SpecImpl&>(*new_spec_impl).value_,
                                  spec_builder);
      return new_spec_impl;
    }

    Spec value_;
  };

  ResourceSpecImplPtr Default() const override {
    return ResourceSpecImplPtr(new SpecImpl(traits_.Default()));
  }

  Result<ResourceSpecImplPtr> FromJson(
      const ::nlohmann::json& j,
      JsonSerializationOptions options) const override {
    auto result =
        internal_json_binding::FromJson<Spec>(j, traits_.JsonBinder(), options);
    if (!result) return std::move(result).status();
    return ResourceSpecImplPtr(new SpecImpl(std::move(*result)));
  }

  void AcquireContextReference(ResourceImplBase& resource) const override {
    traits_.AcquireContextReference(
        static_cast<ResourceImpl&>(resource).value_);
  }
  void ReleaseContextReference(ResourceImplBase& resource) const override {
    traits_.ReleaseContextReference(
        static_cast<ResourceImpl&>(resource).value_);
  }

  ResourceSpecImplPtr DoGetSpec(
      const ResourceImplBase& resource,
      const internal::ContextSpecBuilder& spec_builder) const override {
    return ResourceSpecImplPtr(new SpecImpl(traits_.GetSpec(
        static_cast<const ResourceImpl&>(resource).value_, spec_builder)));
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Traits traits_;
};

/// Globally registers a context resource provider.  This is intended for use by
/// the constructor of `ContextResourceRegistration`, which should be defined as
/// a global variable in order to perform the registration before `main` is
/// called.
///
/// \param provider Non-null pointer to provider.
void RegisterContextResourceProvider(
    std::unique_ptr<const ResourceProviderImplBase> provider);

/// Parses a context resource spec JSON representation for the given provider
/// id.
///
/// \param provider_id Provider id, *must* be registered.
/// \param j JSON representation.  There are no preconditions, but an error is
///     returned if `j` is not a valid representation.  Valid representations
///     are a string (reference to another named resource), and an object
///     (specifying a new resource).  If
///     `options.preserve_bound_context_resources_ == true`, `j` may also be a
///     one-element array containing a string or object, in which case the
///     returned spec is wrapped in a `BuilderResourceSpec` to indicate that it
///     should be immediately bound.
/// \param options JSON serialization options.
/// \checks `provider` is registered.
Result<ResourceSpecImplPtr> ResourceSpecFromJson(
    std::string_view provider_id, const ::nlohmann::json& j,
    JsonSerializationOptions options);

/// Returns the default context resource for a given provider.
///
/// \param provider_id Resource provider id, *must* have been registered.  Only
///     call this function for provider ids known to be valid.
/// \checks `provider_id` has been registered.
/// \returns Non-null pointer to spec.
ResourceOrSpecPtr DefaultResourceSpec(std::string_view provider_id);

/// Obtains a resource from a resource spec.
///
/// This does not take into account `context.bind_partial_`; for that, see
/// overload below.
///
/// \param context Context in which to obtain the resource.
/// \param spec Context resource spec.
/// \param trigger Parent context resource that requested this resource.  For
///     top-level resources, must be null.  For nested resources requested via
///     `ContextResourceCreationContext`, must be non-null.
/// \returns Non-null context resource.
Result<ResourceImplWeakPtr> GetOrCreateResource(ContextImpl& context,
                                                ResourceSpecImplBase& spec,
                                                ResourceContainer* trigger);

/// Obtains a resource from an existing resource or spec.
///
/// The behavior depends on the dynamic type of `resource_or_spec`, and on
/// `context->bind_partial_`.
///
/// 1. If `resource_or_spec` is `nullptr`, `resource` is set to `nullptr`.
///
/// 2. If `resource_or_spec` is already a resource, sets `resource` to
///    `resource_or_spec`.
///
/// 3. If `resource_or_spec` is a resource spec:
///
///    a. If `context->bind_partial_ == true` and `resource_or_spec` is not a
///       `BuilderResourceSpec`, sets `resource` to `resource_or_spec` (i.e. the
///       resource remains unbound).
///
///    b. Otherwise, sets `resource` to the resource corresponding to the spec.
///
/// \param context Context in which to obtain the resource.  Must be non-null.
/// \param resource_or_spec May be context resource spec, existing context
///     resource, or `nullptr`.  `resource` is not modified.  If
///     `resource_or_spec` is already a resource, sets `resource` to
///     `resource_or_spec`.  Otherwise, if `context->bind_partial_`
/// \param resource[out] Set to the resultant resource or spec.
/// \param trigger Parent context resource from which this resource is being
///     requested.  May be null for top-level requests.
/// \dchecks `context != nullptr`
absl::Status GetOrCreateResource(ContextImpl* context,
                                 ResourceOrSpecTaggedPtr resource_or_spec,
                                 ResourceContainer* trigger,
                                 ResourceOrSpecPtr& resource);

/// Obtains the context resource spec from a context resource or spec, and
/// registers it with `builder`.
///
/// This is used by `ContextSpecBuilder::AddResource`.
///
/// The returned spec must not be used (i.e. converted to JSON) until after the
/// last reference to `builder` is destroyed.
///
/// \param builder Context spec builder, must be non-null.
/// \param resource_or_spec Context resource or spec to convert.  If
///     `resource_or_spec` is a context resource, returns a corresponding
///     context resource spec.  If `resource_or_spec` is a context resource
///     spec, returns a copy with all embedded context resources unbound (in
///     case any were bound).  Additionally, any referenced context resource
///     identifiers are registered with `builder` to ensure any they are not
///     reused for any newly-unbound resources.
/// \dchecks `builder` is not null.
ResourceOrSpecPtr AddResourceOrSpec(const internal::ContextSpecBuilder& builder,
                                    ResourceOrSpecTaggedPtr resource_or_spec);

/// If `spec` is non-null, resets it to a default spec for the same provider.
/// If `spec` is null, it remains null.
void StripContext(ResourceOrSpecPtr& spec);

/// Parses a `Context::Resource` JSON representation.
///
/// \param provider_id Provider id, *must* be registered.
absl::Status ResourceSpecFromJsonWithDefaults(
    std::string_view provider_id, const JsonSerializationOptions& options,
    ResourceOrSpecPtr& spec, ::nlohmann::json* j);

/// Converts a `Context::Resource` to its JSON representation.
absl::Status ResourceSpecToJsonWithDefaults(
    const JsonSerializationOptions& options, ResourceOrSpecTaggedPtr spec,
    ::nlohmann::json* j);

template <typename Provider, typename Resource>
struct ResourceJsonBinderImpl {
  absl::Status operator()(std::true_type is_loading,
                          const JsonSerializationOptions& options,
                          Resource* obj, ::nlohmann::json* j) const {
    return internal_context::ResourceSpecFromJsonWithDefaults(
        Provider::id, options, obj->impl_, j);
  }
  absl::Status operator()(std::false_type is_loading,
                          const JsonSerializationOptions& options,
                          const Resource* obj, ::nlohmann::json* j) const {
    return internal_context::ResourceSpecToJsonWithDefaults(
        options, obj->impl_.get(), j);
  }
};

/// Type-erased encode function used by the serializer for
/// `Context::Resource<Provider>` defined in `context.h`.
[[nodiscard]] bool EncodeContextResourceOrSpec(
    serialization::EncodeSink& sink,
    const internal_context::ResourceOrSpecPtr& resource);

/// Type-erased decode function used by the serializer for
/// `Context::Resource<Provider>` defined in `context.h`.
[[nodiscard]] bool DecodeContextResourceOrSpec(
    serialization::DecodeSource& source, std::string_view provider_id,
    internal_context::ResourceOrSpecPtr& resource);

/// Direct serializer for non-null `ContextSpecImplPtr`.  This should normally
/// be used via `IndirectPointerSerializer` or
/// `NonNullIndirectPointerSerializer`.
struct ContextSpecImplPtrNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal_context::ContextSpecImplPtr& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   internal_context::ContextSpecImplPtr& value);
};

/// Direct serializer for non-null `ContextImplPtr`.This should normally
/// be used via `IndirectPointerSerializer` or
/// `NonNullIndirectPointerSerializer`.
struct ContextImplPtrNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal_context::ContextImplPtr& value);

  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   internal_context::ContextImplPtr& value);
};

/// Directly serializes a non-null context resource with dynamic `provider_id`.
/// This should normally be used via `IndirectPointerSerializer` or
/// `NonNullIndirectPointerSerializer`.
struct UntypedContextResourceImplPtrNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal_context::ResourceImplWeakPtr& value);
  [[nodiscard]] static bool Decode(
      serialization::DecodeSource& source,
      internal_context::ResourceImplWeakPtr& value);
};

}  // namespace internal_context
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_context::ContextSpecImplPtr)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_context::ContextImplPtr)

#endif  // TENSORSTORE_CONTEXT_IMPL_BASE_H_
