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

#include "tensorstore/context.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_context {

ContextResourceProviderImplBase::~ContextResourceProviderImplBase() = default;
ContextResourceImplBase::~ContextResourceImplBase() = default;
ContextResourceSpecImplBase::~ContextResourceSpecImplBase() = default;

void ContextResourceImplWeakPtrTraits::increment(ContextResourceImplBase* p) {
  intrusive_ptr_increment(p);
}

void ContextResourceImplWeakPtrTraits::decrement(ContextResourceImplBase* p) {
  intrusive_ptr_decrement(p);
}

void ContextResourceImplStrongPtrTraits::increment(ContextResourceImplBase* p) {
  intrusive_ptr_increment(p);
  p->spec_->provider_->AcquireContextReference(p);
}

void ContextResourceImplStrongPtrTraits::decrement(ContextResourceImplBase* p) {
  p->spec_->provider_->ReleaseContextReference(p);
  intrusive_ptr_decrement(p);
}

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

void intrusive_ptr_increment(ContextSpecImpl* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<ContextSpecImpl>*>(p));
}

void intrusive_ptr_decrement(ContextSpecImpl* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<ContextSpecImpl>*>(p));
}

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

void intrusive_ptr_increment(ContextImpl* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<ContextImpl>*>(p));
}

void intrusive_ptr_decrement(ContextImpl* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<ContextImpl>*>(p));
}

namespace {
struct ContextProviderRegistry {
  struct ProviderKey : public absl::string_view {
    using Base = absl::string_view;
    ProviderKey(const std::unique_ptr<const ContextResourceProviderImplBase>& p)
        : Base(p->id_) {}
    ProviderKey(absl::string_view s) : Base(s) {}
  };

  struct ProviderHash : public absl::Hash<ProviderKey> {
    using is_transparent = void;
  };

  struct ProviderEqualTo : public std::equal_to<ProviderKey> {
    using is_transparent = void;
  };

  absl::Mutex mutex_;
  absl::flat_hash_set<std::unique_ptr<const ContextResourceProviderImplBase>,
                      ProviderHash, ProviderEqualTo>
      providers_ ABSL_GUARDED_BY(mutex_);
};

static ContextProviderRegistry& GetRegistry() {
  static internal::NoDestructor<ContextProviderRegistry> registrar;
  return *registrar;
}

/// Checks for a cycle in the `creation_blocked_on_` pointers starting from
/// `container`.  If one is found, returns a node in the cycle.  Note that
/// `container` may lead to a cycle without being part of one.
ContextResourceContainer* FindCycle(ContextResourceContainer* container) {
  // Brent's algorithm for cycle detection.
  std::size_t power = 1;
  std::size_t lambda = 1;
  auto* tortoise = container;
  auto* hare = container->creation_blocked_on_;
  while (true) {
    if (!hare) return nullptr;
    if (tortoise == hare) return tortoise;
    if (power == lambda) {
      // Increase power of 2.
      tortoise = hare;
      power *= 2;
      lambda = 0;
    }
    hare = hare->creation_blocked_on_;
    lambda += 1;
  }
}

void KillCycle(ContextResourceContainer* container) {
  std::vector<std::string> parts;
  auto* node = container;
  do {
    assert(node->spec_);
    parts.push_back(QuoteString(node->spec_->key_));
    node = node->creation_blocked_on_;
  } while (node != container);
  std::reverse(parts.begin(), parts.end());
  auto error = absl::InvalidArgumentError("Context resource reference cycle: " +
                                          absl::StrJoin(parts, " -> "));
  do {
    auto* next = std::exchange(node->creation_blocked_on_, nullptr);
    node->result_ = error;
    node->condvar_.SignalAll();
    node = next;
  } while (node != container);
}

void WaitForCompletion(absl::Mutex* mutex, ContextResourceContainer* container,
                       ContextResourceContainer* trigger) {
  if (trigger) {
    assert(!trigger->creation_blocked_on_);
    trigger->creation_blocked_on_ = container;
  }
  if (!container->ready()) {
    container->condvar_.WaitWithTimeout(mutex, absl::Milliseconds(5));
    if (!container->ready()) {
      if (auto* cycle_node = FindCycle(container)) {
        KillCycle(cycle_node);
      }
      while (!container->ready()) {
        container->condvar_.Wait(mutex);
      }
    }
  }
  if (trigger) {
    trigger->creation_blocked_on_ = nullptr;
  }
}

Result<ContextResourceImplStrongPtr> CreateResource(
    ContextImpl* context, ContextResourceSpecImplBase* spec,
    ContextResourceContainer* trigger) {
  std::unique_ptr<ContextResourceContainer> container(
      new ContextResourceContainer);
  auto* container_ptr = container.get();
  container->spec_.reset(spec);
  if (trigger) {
    assert(!trigger->creation_blocked_on_);
    trigger->creation_blocked_on_ = container.get();
  }
  context->resources_.insert(std::move(container));
  {
    internal::ScopedMutexUnlock unlock(&context->root_->mutex_);
    container_ptr->result_ = spec->CreateResource({context, container_ptr});
  }
  if (trigger) {
    trigger->creation_blocked_on_ = nullptr;
  }
  container_ptr->condvar_.SignalAll();
  return container_ptr->result_;
}

}  // namespace

Result<ContextResourceImplWeakPtr> GetResource(
    ContextImpl* context, ContextResourceSpecImplBase* spec,
    ContextResourceContainer* trigger) {
  assert(spec);
  assert(context);
  if (!spec->provider_) {
    // The provider was not linked in, despite linking in code that depends on
    // it.  This indicates a build configuration error.
    TENSORSTORE_LOG_FATAL("Context resource provider not registered for: ",
                          QuoteString(spec->key_));
  }
  const absl::string_view key = spec->key_;
  return ChainResult(
      [&] {
        if (key.empty()) {
          // Inline resource, not memoized.

          // Temporary ContextResourceContainer for deadlock detection.
          ContextResourceContainer container;
          container.spec_.reset(spec);
          if (trigger) {
            absl::MutexLock lock(&context->root_->mutex_);
            assert(!trigger->creation_blocked_on_);
            trigger->creation_blocked_on_ = &container;
          }
          auto result = spec->CreateResource({context, &container});
          if (trigger) {
            absl::MutexLock lock(&context->root_->mutex_);
            trigger->creation_blocked_on_ = nullptr;
          }
          return result;
        }
        absl::MutexLock lock(&context->root_->mutex_);
        assert(context->spec_);
#ifndef NDEBUG
        {
          auto it = context->spec_->resources_.find(key);
          assert(it != context->spec_->resources_.end() && it->get() == spec);
        }
#endif
        if (auto it = context->resources_.find(key);
            it != context->resources_.end()) {
          auto* container = it->get();
          WaitForCompletion(&context->root_->mutex_, container, trigger);
          return container->result_;
        }
        return CreateResource(context, spec, trigger);
      }(),
      [](ContextResourceImplStrongPtr p) {
        p->spec_->provider_->ReleaseContextReference(p.get());
        return ContextResourceImplWeakPtr(p.release(),
                                          internal::adopt_object_ref);
      });
}

class ContextResourceReference : public ContextResourceSpecImplBase {
 public:
  ContextResourceReference(const std::string& referent) : referent_(referent) {}

  Result<ContextResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override {
    // Look up referent.
    absl::string_view referent = referent_;
    auto* mutex = &creation_context.context_->root_->mutex_;
    absl::MutexLock lock(mutex);
    ContextImpl* c = creation_context.context_;
    if (referent.empty()) {
      // Refers to default value in parent.  Only valid within a context spec.
      assert(!key_.empty());
      if (c->parent_) {
        c = c->parent_.get();
        referent = provider_->id_;
      } else {
        // Create new default value.
        auto default_spec = provider_->Default();
        default_spec->provider_ = provider_;
        default_spec->key_ = key_;
        default_spec->is_default_ = true;
        return internal_context::CreateResource(c, default_spec.get(),
                                                creation_context.trigger_);
      }
    }
    while (true) {
      if (auto it = c->resources_.find(referent); it != c->resources_.end()) {
        WaitForCompletion(mutex, it->get(), creation_context.trigger_);
        return it->get()->result_;
      }
      auto* context_spec = c->spec_.get();
      if (context_spec) {
        if (auto it = context_spec->resources_.find(referent);
            it != context_spec->resources_.end()) {
          return internal_context::CreateResource(c, it->get(),
                                                  creation_context.trigger_);
        }
      }
      if (!c->parent_) {
        if (referent != provider_->id_) {
          return absl::InvalidArgumentError(
              StrCat("Resource not defined: ", QuoteString(referent)));
        }
        // Create default.
        auto default_spec = provider_->Default();
        default_spec->provider_ = provider_;
        default_spec->key_ = std::string(provider_->id_);
        default_spec->is_default_ = true;
        return internal_context::CreateResource(c, default_spec.get(),
                                                creation_context.trigger_);
      }
      c = c->parent_.get();
    }
  }

  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override {
    if (referent_.empty()) return nullptr;
    return referent_;
  }
  std::string referent_;
};

void RegisterContextResourceProvider(
    std::unique_ptr<const ContextResourceProviderImplBase> provider) {
  auto& registry = GetRegistry();
  absl::MutexLock lock(&registry.mutex_);
  auto id = provider->id_;
  if (!registry.providers_.insert(std::move(provider)).second) {
    TENSORSTORE_LOG_FATAL("Provider ", QuoteString(id), " already registered");
  }
}

const ContextResourceProviderImplBase* GetProvider(absl::string_view id) {
  auto& registry = GetRegistry();
  absl::ReaderMutexLock lock(&registry.mutex_);
  auto it = registry.providers_.find(id);
  if (it == registry.providers_.end()) return nullptr;
  return it->get();
}

namespace {

Status ProviderNotRegisteredError(absl::string_view key) {
  return absl::InvalidArgumentError(
      StrCat("Invalid context resource identifier: ", QuoteString(key)));
}

class UnknownContextResource : public ContextResourceSpecImplBase {
 public:
  UnknownContextResource(const ::nlohmann::json& j) : json_spec_(j) {}

  Result<ContextResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override {
    // This should be unreachable.
    TENSORSTORE_LOG_FATAL("Provider not registered: ", QuoteString(key_));
  }

  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override {
    return json_spec_;
  }

  ::nlohmann::json json_spec_;
};

// Returns the provider.
absl::string_view ParseResourceProvider(absl::string_view key) {
  return key.substr(0, key.find('#'));
}

Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJson(
    const ContextResourceProviderImplBase& provider, const ::nlohmann::json& j,
    ContextFromJsonOptions options) {
  ContextResourceSpecImplPtr impl;
  if (j.is_null()) {
    // Refers to default value in parent.
    impl.reset(new ContextResourceReference(""));
  } else if (auto* s = j.get_ptr<const std::string*>()) {
    auto provider_id = ParseResourceProvider(*s);
    if (provider_id != provider.id_) {
      return absl::InvalidArgumentError(StrCat("Invalid reference to ",
                                               QuoteString(provider.id_),
                                               " resource: ", QuoteString(*s)));
    }
    impl.reset(new ContextResourceReference(*s));
  } else {
    if (!j.is_object()) {
      return internal_json::ExpectedError(j, "string or object");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(impl, provider.FromJson(j, options));
  }
  impl->provider_ = &provider;
  return impl;
}

Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJsonWithKey(
    absl::string_view key, const ::nlohmann::json& j,
    Context::FromJsonOptions options) {
  auto* provider = GetProvider(ParseResourceProvider(key));
  ContextResourceSpecImplPtr impl;
  if (!provider) {
    if (!options.allow_unregistered()) return ProviderNotRegisteredError(key);
    impl.reset(new UnknownContextResource(j));
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        impl, ContextResourceSpecFromJson(*provider, j, options));
  }
  impl->key_.assign(key.begin(), key.end());
  return impl;
}

}  // namespace

Result<ContextResourceSpecImplPtr> ContextResourceSpecFromJson(
    absl::string_view provider_id, const ::nlohmann::json& j,
    Context::FromJsonOptions options) {
  auto* provider = GetProvider(provider_id);
  if (!provider) {
    // Indicates a build configuration problem.
    TENSORSTORE_LOG_FATAL("Context resource provider ",
                          QuoteString(provider_id), " not registered");
  }
  if (j.is_null()) {
    return internal_json::ExpectedError(j, "string or object");
  }
  return ContextResourceSpecFromJson(*provider, j, options);
}

ContextResourceSpecImplPtr DefaultContextResourceSpec(
    absl::string_view provider_id) {
  return ContextResourceSpecFromJson(provider_id, std::string(provider_id), {})
      .value();
}

}  // namespace internal_context

Context Context::Default() {
  Context context;
  context.impl_.reset(new internal_context::ContextImpl);
  context.impl_->root_ = context.impl_.get();
  return context;
}

Context::Context(const Context::Spec& spec, Context parent)
    : impl_(new internal_context::ContextImpl) {
  impl_->spec_ = spec.impl_;
  impl_->parent_ = std::move(parent.impl_);
  if (impl_->parent_) {
    impl_->root_ = impl_->parent_->root_;
  } else {
    impl_->root_ = impl_.get();
  }
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Context::Spec, [](auto is_loading,
                                                         const auto& options,
                                                         auto* obj, auto* j) {
  if constexpr (!is_loading) {
    if (!options.include_context()) {
      *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
      return absl::OkStatus();
    }
  }
  namespace jb = tensorstore::internal::json_binding;
  return jb::DefaultValue(
      [](auto* obj) { *obj = Context::Spec(); },
      jb::Compose<::nlohmann::json::object_t>([](auto is_loading,
                                                 const auto& options, auto* obj,
                                                 auto* j_obj) -> Status {
        if constexpr (is_loading) {
          obj->impl_.reset(new internal_context::ContextSpecImpl);
          obj->impl_->resources_.reserve(j_obj->size());

          for (const auto& [key, value] : *j_obj) {
            TENSORSTORE_ASSIGN_OR_RETURN(
                auto resource,
                internal_context::ContextResourceSpecFromJsonWithKey(key, value,
                                                                     options));
            obj->impl_->resources_.insert(std::move(resource));
          }
        } else {
          if (!obj->impl_) return absl::OkStatus();
          for (const auto& resource_spec : obj->impl_->resources_) {
            if (!options.include_defaults() && resource_spec->is_default_ &&
                resource_spec->key_ == resource_spec->provider_->id_) {
              continue;
            }
            TENSORSTORE_ASSIGN_OR_RETURN(auto resource_spec_json,
                                         resource_spec->ToJson(options));
            j_obj->emplace(resource_spec->key_, std::move(resource_spec_json));
          }
        }
        return absl::OkStatus();
      }))(is_loading, options, obj, j);
})

namespace internal_context {

class BuilderResourceSpec : public ContextResourceSpecImplBase {
 public:
  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override {
    if (!underlying_spec_->key_.empty()) return underlying_spec_->key_;
    return underlying_spec_->ToJson(options);
  }
  Result<ContextResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override {
    return underlying_spec_->CreateResource(creation_context);
  }
  ContextResourceSpecImplPtr underlying_spec_;
};

class BuilderImpl : public internal::AtomicReferenceCount<BuilderImpl> {
 public:
  struct ResourceEntry {
    internal::IntrusivePtr<BuilderResourceSpec> spec;
    bool shared = false;
    size_t id;
  };

  ~BuilderImpl() {
    absl::flat_hash_map<std::string, std::size_t> ids;
    using SharedEntry = std::pair<ContextResourceImplBase*, ResourceEntry*>;
    std::vector<SharedEntry> shared_entries;
    for (auto& p : resources_) {
      absl::string_view key = p.first->spec_->key_;
      if (!key.empty()) {
        ids[key]++;
      }
      if (p.second.shared) {
        shared_entries.emplace_back(p.first.get(), &p.second);
      }
    }
    // Sort by order of insertion to ensure deterministic result.
    std::sort(shared_entries.begin(), shared_entries.end(),
              [](const SharedEntry& a, const SharedEntry& b) {
                return a.second->id < b.second->id;
              });
    for (auto [resource, entry] : shared_entries) {
      std::string key = resource->spec_->key_;
      if (key.empty() || ids.at(key) != 1) {
        // Find the first number `i` such that `<id>#<i>` is unused.
        size_t i = 0;
        while (true) {
          key = StrCat(resource->spec_->provider_->id_, "#", i);
          if (!ids.count(key)) break;
          ++i;
        }
        ids[key]++;
      }
      // Leave `entry->spec->key_` unset since `entry->spec` is exposed as a
      // `Context::Spec`
      entry->spec->underlying_spec_->key_ = key;
      root_->resources_.insert(entry->spec->underlying_spec_);
    }
  }

  ContextSpecImplPtr root_;
  size_t next_id_ = 0;
  absl::flat_hash_map<ContextResourceImplWeakPtr, ResourceEntry> resources_;
};

void intrusive_ptr_increment(BuilderImpl* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<BuilderImpl>*>(p));
}

void intrusive_ptr_decrement(BuilderImpl* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<BuilderImpl>*>(p));
}
}  // namespace internal_context

namespace internal {
ContextSpecBuilder ContextSpecBuilder::Make(ContextSpecBuilder parent) {
  ContextSpecBuilder builder;
  if (parent.builder_impl_) {
    builder.builder_impl_ = std::move(parent.builder_impl_);
  } else {
    builder.builder_impl_.reset(new internal_context::BuilderImpl);
    builder.spec_impl_.reset(new internal_context::ContextSpecImpl);
    builder.builder_impl_->root_ = builder.spec_impl_;
  }
  return builder;
}

Context::Spec ContextSpecBuilder::spec() const {
  Context::Spec spec;
  spec.impl_ = spec_impl_;
  return spec;
}

internal_context::ContextResourceSpecImplPtr ContextSpecBuilder::AddResource(
    internal_context::ContextResourceImplBase* resource) const {
  internal_context::ContextResourceImplWeakPtr resource_ptr(resource);
  auto& entry = builder_impl_->resources_[resource_ptr];
  if (!entry.spec) {
    // Register new resource.
    entry.spec.reset(new internal_context::BuilderResourceSpec);
    entry.spec->provider_ = resource->spec_->provider_;
    entry.id = builder_impl_->next_id_++;
    entry.spec->underlying_spec_ =
        resource->spec_->provider_->GetSpec(resource, *this);
    entry.shared =
        (resource->spec_->is_default_ || !resource->spec_->key_.empty());
  } else {
    entry.shared = true;
  }
  return entry.spec;
}

}  // namespace internal

}  // namespace tensorstore
