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

#include <nlohmann/json.hpp>
#include "tensorstore/context_impl_base.h"
#include "tensorstore/internal/cache_key.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"

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
/// mutable state, such as a cache.  Resources are constructed lazily from a
/// `ResourceSpec` (which is created from a JSON representation) when first
/// accessed, and the `Context::Resource` handle retains the named identifier
///
/// Context resources are used with the TensorStore library to specify
/// TensorStore driver and KeyValueStore driver configuration options that do
/// not affect the identity of the KeyValueStore or TensorStore; in general, the
/// contents of a KeyValueStore or TensorStore should be identical for any valid
/// context.  For example, a file or bucket path for accessing data should not
/// be specified as a context resource, while authentication credentials,
/// caching, and concurrency options should be specified as context resources.
///
/// Example usage:
///
///     auto spec_result = Context::Spec::FromJson({
///       {"data_copy_concurrency", {{"limit", 4}}},
///       {"data_copy_concurrency#a", {{"limit", 8}}},
///     });
///     if (!spec_result) {
///       // Handle error
///     }
///     auto resource_spec_result =
///         Context::ResourceSpec<internal::DataCopyConcurrencyResource>
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
///         Context::ResourceSpec<internal::DataCopyConcurrencyResource>
///         ::FromJson("data_copy_concurrency#a").value()).value();
///     resource2->executor([] { /* task body */ });
///
/// `Context::Spec` and `Context::ResourceSpec` serve as intermediate
/// representations between JSON and the actual context and context resource,
/// respectively.  This indirection makes it possible to validate a context or
/// resource specification without actually constructing the resource; this is
/// especially useful if one process/machine prepares a specification for use by
/// another process/machine.  It also makes it possible to properly handle
/// shared resources when serializing back to JSON (see
/// `internal::ContextSpecBuilder`).
///
/// \threadsafety Thread compatible.
class Context {
 public:
  using ToJsonOptions = ContextToJsonOptions;
  using FromJsonOptions = ContextFromJsonOptions;
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

  /// Specifies a context resource either directly or by named reference.
  ///
  /// This may be used to obtain a `Resource` from a `Context` object.  Named
  /// references are not resolved until the `Resource` is created.
  template <typename Provider>
  class ResourceSpec
      : public internal::json_binding::JsonBindable<
            ResourceSpec<Provider>, FromJsonOptions, ToJsonOptions> {
   public:
    /// Constructs an invalid resource spec.
    ResourceSpec() = default;

    /// Returns `true` if this represents a valid resource spec.
    explicit operator bool() const { return static_cast<bool>(impl_); }

    /// Returns a resource spec that refers to the default value within the
    /// `Context`.
    ///
    /// The returned `ResourceSpec` is guaranteed to be valid.
    static ResourceSpec<Provider> Default() {
      ResourceSpec<Provider> r;
      r.impl_ = internal_context::DefaultContextResourceSpec(Provider::id);
      return r;
    }

    static constexpr auto default_json_binder =
        internal::json_binding::DefaultValue(
            [](auto* obj) {
              // Use dependent syntax `obj->Default()` rather than `Default()`
              // because `ResourceSpec` is still incomplete.
              *obj = obj->Default();
              return absl::OkStatus();
            },
            [](auto is_loading, const auto& options, auto* obj,
               auto* j) -> Status {
              if constexpr (is_loading) {
                TENSORSTORE_ASSIGN_OR_RETURN(
                    obj->impl_, internal_context::ContextResourceSpecFromJson(
                                    Provider::id, std::move(*j), options));
              } else {
                if (!options.include_context() || !obj->impl_) {
                  *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
                  return absl::OkStatus();
                }
                TENSORSTORE_ASSIGN_OR_RETURN(*j, obj->impl_->ToJson(options));
              }
              return absl::OkStatus();
            });

   private:
    friend class Context;
    friend class internal::ContextSpecBuilder;
    internal_context::ContextResourceSpecImplPtr impl_;
  };

  /// Shared handle to a context resource.
  ///
  /// Valid handles are obtained from a `ResourceSpec` object using
  /// `Context::GetResource`.
  template <typename Provider>
  class Resource {
    using ResourceImpl = internal_context::ContextResourceImpl<Provider>;

   public:
    using element_type = const typename Provider::Resource;

    /// Constructs an invalid handle.
    Resource() = default;

    /// Returns a pointer to the resource object, or `nullptr` if this is an
    /// invalid handle.
    element_type* get() const noexcept {
      return impl_ ? &(static_cast<ResourceImpl*>(impl_.get())->value_)
                   : nullptr;
    }
    element_type* operator->() const noexcept { return get(); }
    element_type& operator*() const noexcept { return *get(); }

    /// Returns `true` if this is a valid handle.
    explicit operator bool() const { return static_cast<bool>(impl_); }

    friend void EncodeCacheKeyAdl(std::string* out, const Resource& resource) {
      internal::EncodeCacheKey(
          out, reinterpret_cast<std::uintptr_t>(resource.get()));
    }

   private:
    friend class Context;
    friend class internal::ContextSpecBuilder;
    internal_context::ContextResourceImplWeakPtr impl_;
  };

  /// Constructs a null context.
  Context() = default;

  /// Constructs a new default context.
  static Context Default();

  /// Constructs a context from a `ContextSpec`.
  ///
  /// The actual resources are created lazily as needed.
  ///
  /// \param spec The context spec.
  /// \param parent The parent context to extend.  Specifying a null context is
  ///     equivalent to specifying `Default()`.
  explicit Context(const Spec& spec, Context parent = {});

  /// Returns a resource.
  ///
  /// \param resource_spec The resource spec.
  /// \threadsafety Thread safe.
  template <typename Provider>
  Result<Resource<Provider>> GetResource(
      const ResourceSpec<Provider>& resource_spec) const {
    Resource<Provider> resource;
    TENSORSTORE_ASSIGN_OR_RETURN(
        resource.impl_, internal_context::GetResource(
                            impl_.get(), resource_spec.impl_.get(), nullptr));
    return resource;
  }

  /// Returns `true` if this is not a null context.
  explicit operator bool() const { return static_cast<bool>(impl_); }

 private:
  friend class internal_context::Access;
  internal_context::ContextImplPtr impl_;
};

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
  static ContextSpecBuilder Make(ContextSpecBuilder parent = {});

  /// Registers a resource with this builder, and returns a `ResourceSpec` that
  /// refers to it.
  ///
  /// If `resource` is shared (meaning this method is called multiple times
  /// within the same `ContextSpecBuilder` tree with the same `resource`) or if
  /// `resource` was constructed from a named context resource spec, the
  /// returned `ResourceSpec` will be a named reference to the actual resource
  /// specification defined in the `Context::Spec` returned by `spec()`.
  ///
  /// Otherwise, the returned `ResourceSpec` will specify the resource directly.
  ///
  /// The returned `ResourceSpec` must not be used until all
  /// `ContextSpecBuilder` objects within the tree have been destroyed.
  template <typename Provider>
  Context::ResourceSpec<Provider> AddResource(
      const Context::Resource<Provider>& resource) const {
    assert(impl_);
    Context::ResourceSpec<Provider> resource_spec;
    resource_spec.impl_ =
        internal_context::AddResource(*this, resource.impl_.get());
    return resource_spec;
  }

  /// Returns the `Context::Spec` that defines the shared resources registered
  /// with this builder or a descendant builder.
  Context::Spec spec() const;

 private:
  friend class internal_context::Access;
  internal_context::BuilderImplPtr impl_;
  internal_context::ContextSpecImplPtr spec_impl_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_CONTEXT_H_
