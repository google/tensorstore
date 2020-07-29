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
#include "absl/container/flat_hash_set.h"
#include "tensorstore/context.h"

namespace tensorstore {
namespace internal_context {

/// Implementation of `Context::Spec`.  Maintains a set of
/// `ContextResourceSpecImpl` objects, corresponding to `Context::ResourceSpec`
/// objects in the public interface.
class ContextSpecImpl : public internal::AtomicReferenceCount<ContextSpecImpl> {
 public:
  /// Helper type to support heterogeneous lookup by
  /// `ContextResourceSpecImplBase` pointer or by key.
  struct ResourceSpecKey : public absl::string_view {
    using Base = absl::string_view;
    ResourceSpecKey(const ContextResourceSpecImplPtr& p) : Base(p->key_) {}
    ResourceSpecKey(absl::string_view s) : Base(s) {}
  };

  struct ResourceSpecKeyHash : public absl::Hash<ResourceSpecKey> {
    using is_transparent = void;
  };

  struct ResourceSpecKeyEqualTo : public std::equal_to<ResourceSpecKey> {
    using is_transparent = void;
  };

  absl::flat_hash_set<ContextResourceSpecImplPtr, ResourceSpecKeyHash,
                      ResourceSpecKeyEqualTo>
      resources_;
};

/// Holder for a `ContextResourceBaseImpl` object within `ContextImpl`, used for
/// waiting while it is being constructed, and retained afterwards simply to
/// reference the constructed value.
class ContextResourceContainer {
 public:
  ContextResourceSpecImplPtr spec_;
  absl::CondVar condvar_;
  // Specifies that the creation of this resource is blocked on the creation of
  // another resource.  Used to detect deadlock.
  ContextResourceContainer* creation_blocked_on_ = nullptr;

  // Set to non-null when resource is ready or an error has occurred.
  Result<ContextResourceImplStrongPtr> result_ = ContextResourceImplStrongPtr();

  bool ready() const { return !result_ || result_->get() != nullptr; }
};

/// Implementation of `Context`.  Maintains a set of `ContextResourceContainer`
/// objects.
class ContextImpl : public internal::AtomicReferenceCount<ContextImpl> {
 public:
  // To create for a given resource spec:
  //
  //    if resource spec is not a reference, create but don't store result in
  //    the context
  //
  //    if resource spec is a reference, traverse upwards until we
  //       (a) find existing resource   or (b) find spec.
  //       If we don't find spec,  create in root.
  ContextSpecImplPtr spec_;
  ContextImplPtr parent_;
  ContextImpl* root_;

  // Only used in the root context.
  absl::Mutex mutex_;

  /// Key type for the `resources_` hash table that supports heterogeneous
  /// lookup by string key.
  ///
  /// The `resources_` hash table defined below holds
  /// `std::unique_ptr<ContextResourceContainer>` objects (to allow
  /// `ContextResourceContainer` pointers to remain valid despite other
  /// modifications to the hash table).  The `spec_->key_` member of the
  /// `ContextResourceContainer` object serves as the key.
  struct ResourceKey : public absl::string_view {
    using Base = absl::string_view;
    ResourceKey(const std::unique_ptr<ContextResourceContainer>& p)
        : Base(p->spec_->key_) {}
    ResourceKey(const ContextResourceContainer* p) : Base(p->spec_->key_) {}
    ResourceKey(absl::string_view s) : Base(s) {}
  };

  struct ResourceKeyHash : public absl::Hash<ResourceKey> {
    using is_transparent = void;
  };

  struct ResourceKeyEqualTo : public std::equal_to<ResourceKey> {
    using is_transparent = void;
  };

  absl::flat_hash_set<std::unique_ptr<ContextResourceContainer>,
                      ResourceKeyHash, ResourceKeyEqualTo>
      resources_;
};

/// Derived implementation of `ContextResourceSpecImplBase` used by
/// `ContextSpecBuilder`.  This adds a layer of indirection, which allows
/// `ContextSpecBuilder` to return a pointer to a `BuilderResourceSpec` object
/// as each resource is registered and then update the `underlying_spec_` values
/// later after all resources have been added.
class BuilderResourceSpec : public ContextResourceSpecImplBase {
 public:
  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override;
  Result<ContextResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override;
  ContextResourceSpecImplPtr underlying_spec_;
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
  absl::flat_hash_map<ContextResourceImplWeakPtr, ResourceEntry> resources_;
};

/// Returns the provider for a given registered provider `id`, or `nullptr` if
/// `id` has not been registered.
const ContextResourceProviderImplBase* GetProvider(std::string_view id);

/// Given a `key` of the form `<provider-id>#tag`, returns the `<provider-id>`
/// portion.
std::string_view ParseResourceProvider(std::string_view key);

/// Returns an error indicating that the specified context resource key does not
/// specify a known provider.
Status ProviderNotRegisteredError(std::string_view key);

/// Converts a JSON spec to an anonymous context resource spec.
Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJson(
    const ContextResourceProviderImplBase& provider, const ::nlohmann::json& j,
    ContextFromJsonOptions options);

/// Converts a JSON spec to a keyed context resource spec.  The provider is
/// determined from the `key`.
Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJsonWithKey(
    absl::string_view key, const ::nlohmann::json& j,
    Context::FromJsonOptions options);

}  // namespace internal_context
}  // namespace tensorstore

#endif  // TENSORSTORE_CONTEXT_IMPL_H_
