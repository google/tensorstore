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

#ifndef TENSORSTORE_CONTEXT_IMPL_H_
#define TENSORSTORE_CONTEXT_IMPL_H_

/// \file
///
/// Context implementation details not required by `context.h`, but required by
/// `context.cc` and by the Python Context bindings.

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/heterogeneous_container.h"

namespace tensorstore {
namespace internal_context {

/// Implementation of `Context::Spec`.  Maintains a set of `ResourceSpecImpl`
/// objects, corresponding to `Context::ResourceSpec` objects in the public
/// interface.
class ContextSpecImpl : public internal::AtomicReferenceCount<ContextSpecImpl> {
 public:
  internal::HeterogeneousHashSet<ResourceSpecImplPtr, std::string_view,
                                 &ResourceSpecImplBase::key_>
      resources_;
};

/// Holder for a `ResourceBaseImpl` object within `ContextImpl`, used for
/// waiting while it is being constructed, and retained afterwards simply to
/// reference the constructed value.
class ResourceContainer {
 public:
  ResourceSpecImplPtr spec_;
  absl::CondVar condvar_;
  // Specifies that the creation of this resource is blocked on the creation of
  // another resource.  Used to detect deadlock.
  ResourceContainer* creation_blocked_on_ = nullptr;

  // Set to non-null when resource is ready or an error has occurred.
  Result<ResourceImplStrongPtr> result_ = ResourceImplStrongPtr();

  bool ready() const { return !result_ || result_->get() != nullptr; }

  std::string_view spec_key() const { return spec_->key_; }
};

/// Implementation of `Context`.  Maintains a set of `ResourceContainer`
/// objects.
class ContextImpl : public internal::AtomicReferenceCount<ContextImpl> {
 public:
  ContextImpl();
  ~ContextImpl();

  ContextSpecImplPtr spec_;
  ContextImplPtr parent_;
  ContextImpl* root_;

  // Only used in the root context.
  absl::Mutex mutex_;

  // Holds `std::unique_ptr<ResourceContainer>` objects (to allow
  // `ResourceContainer` pointers to remain valid despite other modifications to
  // the hash table).
  internal::HeterogeneousHashSet<std::unique_ptr<ResourceContainer>,
                                 std::string_view, &ResourceContainer::spec_key>
      resources_;

  /// Used in conjunction with
  /// `JsonSerializationOptions::preserve_bound_context_resources_` to indicate
  /// that only previously-bound context resources (wrapped in a
  /// `BuilderResourceSpec`) should be bound.  Other context resource specs
  /// should remain unbound.
  bool bind_partial_ = false;
};

/// Derived implementation of `ResourceSpecImplBase` used by
/// `ContextSpecBuilder`.  This adds a layer of indirection, which allows
/// `ContextSpecBuilder` to return a pointer to a `BuilderResourceSpec` object
/// as each resource is registered and then update the `underlying_spec_` values
/// later after all resources have been added.
class BuilderResourceSpec : public ResourceSpecImplBase {
 public:
  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override;
  Result<ResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override;
  ResourceSpecImplPtr UnbindContext(
      const internal::ContextSpecBuilder& spec_builder) final;

  ResourceSpecImplPtr underlying_spec_;
};

/// Implementation of `ContextSpecBuilder`.
class BuilderImpl : public internal::AtomicReferenceCount<BuilderImpl> {
 public:
  struct ResourceEntry {
    internal::IntrusivePtr<BuilderResourceSpec> spec;
    bool shared = false;
    size_t id;
  };

  ~BuilderImpl();

  ContextSpecImplPtr root_;
  size_t next_id_ = 0;
  absl::flat_hash_map<ResourceImplWeakPtr, ResourceEntry> resources_;

  /// Tracks the resource identifiers used by existing `Context::Spec` instances
  /// (passed to `ContextSpecBuilder::Make`) that are associated with this
  /// builder, in order to ensure they are not reused.
  absl::flat_hash_map<std::string, std::size_t> ids_;
};

/// Returns the provider for a given registered provider `id`, or `nullptr` if
/// `id` has not been registered.
const ResourceProviderImplBase* GetProvider(std::string_view id);

/// Same as above, but a fatal error if the provider is not registered.
const ResourceProviderImplBase& GetProviderOrDie(std::string_view id);

/// Returns a new copy of the default spec for the given provider with the
/// specified key.
///
/// Note that this returns a new `ResourceSpecImpl` of the type defined
/// by the `provider`.  In contrast, `DefaultResourceSpec` merely returns
/// a `ResourceReference` referring to the value defined under the
/// default key in a parent context.
ResourceSpecImplPtr MakeDefaultResourceSpec(
    const ResourceProviderImplBase& provider, std::string_view key);

/// Given a `key` of the form `<provider-id>#tag`, returns the `<provider-id>`
/// portion.
std::string_view ParseResourceProvider(std::string_view key);

/// Returns an error indicating that the specified context resource key does not
/// specify a known provider.
absl::Status ProviderNotRegisteredError(std::string_view key);

/// Converts a JSON spec to an anonymous context resource spec.
Result<ResourceSpecImplPtr> ResourceSpecFromJson(
    const ResourceProviderImplBase& provider, const ::nlohmann::json& j,
    JsonSerializationOptions options);

/// Converts a JSON spec to a keyed context resource spec.  The provider is
/// determined from the `key`.
Result<ResourceSpecImplPtr> ResourceSpecFromJsonWithKey(
    std::string_view key, const ::nlohmann::json& j,
    Context::FromJsonOptions options);

/// Returns the `ContextImpl` that created `resource`, or `nullptr` if that
/// context has been destroyed.
ContextImplPtr GetCreator(ResourceImplBase& resource);

/// Converts a `ResourceSpecImplPtr` to a `ResourceOrSpecPtr`.
inline ResourceOrSpecPtr ToResourceOrSpecPtr(ResourceSpecImplPtr ptr) {
  return ResourceOrSpecPtr(ResourceOrSpecTaggedPtr(ptr.release(), 1),
                           internal::adopt_object_ref);
}

/// Converts a `ResourceImplPtr` to a `ResourceOrSpecPtr`.
inline ResourceOrSpecPtr ToResourceOrSpecPtr(ResourceImplWeakPtr ptr) {
  return ResourceOrSpecPtr(ResourceOrSpecTaggedPtr(ptr.release(), 0),
                           internal::adopt_object_ref);
}

}  // namespace internal_context
}  // namespace tensorstore

#endif  // TENSORSTORE_CONTEXT_IMPL_H_
