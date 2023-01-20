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

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context_impl.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/heterogeneous_container.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/serialization/json.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_context {

ResourceProviderImplBase::~ResourceProviderImplBase() = default;
ResourceOrSpecBase::~ResourceOrSpecBase() = default;
ResourceImplBase::~ResourceImplBase() = default;
ResourceSpecImplBase::~ResourceSpecImplBase() = default;

ContextImplPtr GetCreator(ResourceImplBase& resource) {
  absl::MutexLock lock(&resource.mutex_);
  auto* creator_ptr = resource.weak_creator_;
  if (!creator_ptr ||
      !internal::IncrementReferenceCountIfNonZero(*creator_ptr)) {
    return {};
  }
  return ContextImplPtr(creator_ptr, internal::adopt_object_ref);
}

void ResourceOrSpecPtrTraits::increment(ResourceOrSpecBase* p) {
  intrusive_ptr_increment(p);
}

void ResourceOrSpecPtrTraits::decrement(ResourceOrSpecBase* p) {
  intrusive_ptr_decrement(p);
}

void ResourceImplWeakPtrTraits::increment(ResourceOrSpecBase* p) {
  intrusive_ptr_increment(p);
}

void ResourceImplWeakPtrTraits::decrement(ResourceOrSpecBase* p) {
  intrusive_ptr_decrement(p);
}

void ResourceImplStrongPtrTraits::increment(ResourceImplBase* p) {
  intrusive_ptr_increment(p);
  p->spec_->provider_->AcquireContextReference(*p);
}

void ResourceImplStrongPtrTraits::decrement(ResourceImplBase* p) {
  p->spec_->provider_->ReleaseContextReference(*p);
  intrusive_ptr_decrement(p);
}

void intrusive_ptr_increment(ContextSpecImpl* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<ContextSpecImpl>*>(p));
}

void intrusive_ptr_decrement(ContextSpecImpl* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<ContextSpecImpl>*>(p));
}

void intrusive_ptr_increment(ContextImpl* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<ContextImpl>*>(p));
}

void intrusive_ptr_decrement(ContextImpl* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<ContextImpl>*>(p));
}

ContextImpl::ContextImpl() = default;
ContextImpl::~ContextImpl() {
  // Invalidate weak references to `this`.
  for (const auto& resource_container : resources_) {
    auto& result = resource_container->result_;
    if (!result.ok()) continue;
    auto& resource = **result;
    absl::MutexLock lock(&resource.mutex_);
    // Only reset the `weak_creator_` if it points to `this`.  `resources_` can
    // contain resources that are actually references to resources in a parent
    // context, in which case we must not change `weak_creator_`.
    if (resource.weak_creator_ == this) {
      resource.weak_creator_ = nullptr;
    }
  }
}

namespace {
struct ContextProviderRegistry {
  absl::Mutex mutex_;
  internal::HeterogeneousHashSet<
      std::unique_ptr<const ResourceProviderImplBase>, std::string_view,
      &ResourceProviderImplBase::id_>
      providers_ ABSL_GUARDED_BY(mutex_);
};

static ContextProviderRegistry& GetRegistry() {
  static internal::NoDestructor<ContextProviderRegistry> registrar;
  return *registrar;
}

// Checks for a cycle in the `creation_blocked_on_` pointers starting from
// `container`.  If one is found, returns a node in the cycle.  Note that
// `container` may lead to a cycle without being part of one.
//
// Must be called with root context mutex held.
ResourceContainer* FindCycle(ResourceContainer* container) {
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

// Must be called with root context mutex held.
void KillCycle(ResourceContainer* container) {
  std::vector<std::string> parts;
  auto* node = container;
  do {
    assert(node->spec_);
    std::string part;
    if (!node->spec_->key_.empty()) {
      tensorstore::StrAppend(&part, QuoteString(node->spec_->key_), ":");
    }
    auto json_result = node->spec_->ToJson(IncludeDefaults{true});
    if (json_result.has_value()) {
      tensorstore::StrAppend(&part, json_result->dump());
    } else {
      tensorstore::StrAppend(
          &part, "unprintable spec for ",
          tensorstore::QuoteString(node->spec_->provider_->id_));
    }
    parts.push_back(std::move(part));
    node = node->creation_blocked_on_;
  } while (node != container);
  auto error = absl::InvalidArgumentError("Context resource reference cycle: " +
                                          absl::StrJoin(parts, " -> "));
  do {
    auto* next = std::exchange(node->creation_blocked_on_, nullptr);
    node->result_ = error;
    node->condvar_.SignalAll();
    node = next;
  } while (node != container);
}

void WaitForCompletion(absl::Mutex* mutex, ResourceContainer* container,
                       ResourceContainer* trigger) {
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

Result<ResourceImplStrongPtr> CreateResource(ContextImpl& context,
                                             ResourceSpecImplBase& spec,
                                             ResourceContainer* trigger) {
  std::unique_ptr<ResourceContainer> container(new ResourceContainer);
  auto* container_ptr = container.get();
  container->spec_.reset(&spec);
  if (trigger) {
    assert(!trigger->creation_blocked_on_);
    trigger->creation_blocked_on_ = container.get();
  }
  context.resources_.insert(std::move(container));
  Result<ResourceImplStrongPtr> result{};
  {
    internal::ScopedWriterUnlock unlock(context.root_->mutex_);
    result = spec.CreateResource({&context, container_ptr});
    if (result.ok()) {
      auto& resource = **result;
      // Set `weak_creator_` if `resource` was created directly from `spec` by
      // `context`.  The alternative is that `spec` is a
      // `ResourceReference` and `resource` was created by a parent
      // context, in which case `weak_creator_` must not be modified.
      if (resource.spec_.get() == &spec) {
        // Resource was created directly from `spec`, which means it was defined
        // directly by `context->spec_`.  At this point, no other thread has
        // access to `resource`, because `trigger->creation_blocked_on_` has not
        // yet been cleared.  Therefore, it is safe to access
        // `resource.weak_creator_` without locking `resource.mutex_`.  However,
        // for consistency we acquire it anyway.
        absl::MutexLock lock(&resource.mutex_);
        assert(resource.weak_creator_ == nullptr);
        resource.weak_creator_ = &context;
      }
    }
  }
  container_ptr->result_ = std::move(result);
  if (trigger) {
    trigger->creation_blocked_on_ = nullptr;
  }
  container_ptr->condvar_.SignalAll();
  return container_ptr->result_;
}

Result<ResourceImplStrongPtr> GetOrCreateResourceStrongPtr(
    ContextImpl& context, ResourceSpecImplBase& spec,
    ResourceContainer* trigger) {
  if (!spec.provider_) {
    // The provider was not linked in, despite linking in code that depends on
    // it.  This indicates a build configuration error.
    ABSL_LOG(FATAL) << "Context resource provider not registered for: "
                    << QuoteString(spec.key_);
  }
  const std::string_view key = spec.key_;
  if (key.empty()) {
    // Inline resource, not memoized.

    // Temporary ResourceContainer for deadlock detection.
    ResourceContainer container;
    container.spec_.reset(&spec);
    if (trigger) {
      absl::MutexLock lock(&context.root_->mutex_);
      assert(!trigger->creation_blocked_on_);
      trigger->creation_blocked_on_ = &container;
    }
    auto result = spec.CreateResource({&context, &container});
    if (trigger) {
      absl::MutexLock lock(&context.root_->mutex_);
      trigger->creation_blocked_on_ = nullptr;
    }
    return result;
  }
  absl::MutexLock lock(&context.root_->mutex_);
  assert(context.spec_);
#ifndef NDEBUG
  {
    auto it = context.spec_->resources_.find(key);
    assert(it != context.spec_->resources_.end() && it->get() == &spec);
  }
#endif
  if (auto it = context.resources_.find(key); it != context.resources_.end()) {
    auto* container = it->get();
    WaitForCompletion(&context.root_->mutex_, container, trigger);
    return container->result_;
  }
  return CreateResource(context, spec, trigger);
}

}  // namespace

Result<ResourceImplWeakPtr> GetOrCreateResource(ContextImpl& context,
                                                ResourceSpecImplBase& spec,
                                                ResourceContainer* trigger) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto p, GetOrCreateResourceStrongPtr(context, spec, trigger));
  p->spec_->provider_->ReleaseContextReference(*p);
  return ResourceImplWeakPtr(p.release(), internal::adopt_object_ref);
}

class ResourceReference : public ResourceSpecImplBase {
 public:
  ResourceReference(const std::string& referent) : referent_(referent) {}

  Result<ResourceImplStrongPtr> CreateResource(
      const internal::ContextResourceCreationContext& creation_context)
      override {
    // Look up referent.
    std::string_view referent = referent_;
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
        auto default_spec = MakeDefaultResourceSpec(*provider_, key_);
        return internal_context::CreateResource(*c, *default_spec,
                                                creation_context.trigger_);
      }
    }
    while (true) {
      if (auto it = c->resources_.find(referent); it != c->resources_.end()) {
        ResourceContainer* container = it->get();
        WaitForCompletion(mutex, container, creation_context.trigger_);
        return container->result_;
      }
      auto* context_spec = c->spec_.get();
      if (context_spec) {
        if (auto it = context_spec->resources_.find(referent);
            it != context_spec->resources_.end()) {
          return internal_context::CreateResource(*c, **it,
                                                  creation_context.trigger_);
        }
      }
      if (!c->parent_) {
        if (referent != provider_->id_) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Resource not defined: ", QuoteString(referent)));
        }
        // Create default.
        auto default_spec = MakeDefaultResourceSpec(*provider_, provider_->id_);
        return internal_context::CreateResource(*c, *default_spec,
                                                creation_context.trigger_);
      }
      c = c->parent_.get();
    }
  }

  Result<::nlohmann::json> ToJson(Context::ToJsonOptions options) override {
    if (referent_.empty()) return nullptr;
    return referent_;
  }

  ResourceSpecImplPtr UnbindContext(
      const internal::ContextSpecBuilder& spec_builder) final {
    auto& builder_impl = *internal_context::Access::impl(spec_builder);
    // Ensure the referent is not re-used as an identifier for another resource.
    ++builder_impl.ids_[referent_];
    return ResourceSpecImplPtr(this);
  }

  std::string referent_;
};

void RegisterContextResourceProvider(
    std::unique_ptr<const ResourceProviderImplBase> provider) {
  auto& registry = GetRegistry();
  absl::MutexLock lock(&registry.mutex_);
  auto id = provider->id_;
  if (!registry.providers_.insert(std::move(provider)).second) {
    ABSL_LOG(FATAL) << "Provider " << QuoteString(id) << " already registered";
  }
}

const ResourceProviderImplBase* GetProvider(std::string_view id) {
  auto& registry = GetRegistry();
  absl::ReaderMutexLock lock(&registry.mutex_);
  auto it = registry.providers_.find(id);
  if (it == registry.providers_.end()) return nullptr;
  return it->get();
}

const ResourceProviderImplBase& GetProviderOrDie(std::string_view id) {
  auto* provider = GetProvider(id);
  if (!provider) {
    // Indicates a build configuration problem.
    ABSL_LOG(FATAL) << "Context resource provider " << QuoteString(id)
                    << " not registered";
  }
  return *provider;
}

ResourceSpecImplPtr MakeDefaultResourceSpec(
    const ResourceProviderImplBase& provider, std::string_view key) {
  auto default_spec = provider.Default();
  default_spec->provider_ = &provider;
  default_spec->key_ = key;
  default_spec->is_default_ = true;
  return default_spec;
}

// Returns the provider id.
std::string_view ParseResourceProvider(std::string_view key) {
  return key.substr(0, key.find('#'));
}

absl::Status ProviderNotRegisteredError(std::string_view key) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Invalid context resource identifier: ", QuoteString(key)));
}

Result<ResourceSpecImplPtr> ResourceSpecFromJson(
    const ResourceProviderImplBase& provider, const ::nlohmann::json& j,
    JsonSerializationOptions options) {
  ResourceSpecImplPtr impl;
  if (j.is_null()) {
    // Refers to default value in parent.
    impl.reset(new ResourceReference(""));
  } else if (auto* s = j.get_ptr<const std::string*>()) {
    auto provider_id = ParseResourceProvider(*s);
    if (provider_id != provider.id_) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Invalid reference to ", QuoteString(provider.id_),
          " resource: ", QuoteString(*s)));
    }
    impl.reset(new ResourceReference(*s));
  } else {
    if (!j.is_object()) {
      return internal_json::ExpectedError(j, "string or object");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(impl, provider.FromJson(j, options));
  }
  impl->provider_ = &provider;
  return impl;
}

Result<ResourceSpecImplPtr> ResourceSpecFromJsonWithKey(
    std::string_view key, const ::nlohmann::json& j,
    Context::FromJsonOptions options) {
  auto* provider = GetProvider(ParseResourceProvider(key));
  ResourceSpecImplPtr impl;
  if (!provider) {
    return ProviderNotRegisteredError(key);
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(impl,
                                 ResourceSpecFromJson(*provider, j, options));
  }
  impl->key_ = key;
  return impl;
}

Result<ResourceSpecImplPtr> ResourceSpecFromJson(
    std::string_view provider_id, const ::nlohmann::json& j,
    Context::FromJsonOptions options) {
  auto& provider = GetProviderOrDie(provider_id);
  if (j.is_null()) {
    return internal_json::ExpectedError(j, "string or object");
  }
  return ResourceSpecFromJson(provider, j, options);
}

ResourceOrSpecPtr DefaultResourceSpec(std::string_view provider_id) {
  return ToResourceOrSpecPtr(
      ResourceSpecFromJson(provider_id, std::string(provider_id), {}).value());
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

Result<Context> Context::FromJson(::nlohmann::json json_spec, Context parent,
                                  FromJsonOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto spec, Spec::FromJson(std::move(json_spec), std::move(options)));
  return Context(spec, std::move(parent));
}

Context::Spec Context::spec() const {
  if (!impl_) return {};
  Context::Spec spec;
  internal_context::Access::impl(spec) = impl_->spec_;
  return spec;
}

Context Context::parent() const {
  if (!impl_) return {};
  Context parent_context;
  parent_context.impl_ = impl_->parent_;
  return parent_context;
}

namespace jb = tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    Context::Spec,
    jb::Compose<::nlohmann::json::object_t>([](auto is_loading,
                                               const auto& options, auto* obj,
                                               auto* j_obj) -> absl::Status {
      if constexpr (is_loading) {
        obj->impl_.reset(new internal_context::ContextSpecImpl);
        obj->impl_->resources_.reserve(j_obj->size());

        for (const auto& [key, value] : *j_obj) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto resource, internal_context::ResourceSpecFromJsonWithKey(
                                 key, value, options));
          obj->impl_->resources_.insert(std::move(resource));
        }
      } else {
        if (!obj->impl_) return absl::OkStatus();
        for (const auto& resource_spec : obj->impl_->resources_) {
          TENSORSTORE_ASSIGN_OR_RETURN(auto resource_spec_json,
                                       resource_spec->ToJson(options));
          assert(!resource_spec_json.is_discarded());
          j_obj->emplace(resource_spec->key_, std::move(resource_spec_json));
        }
      }
      return absl::OkStatus();
    }))

namespace internal {

TENSORSTORE_DEFINE_JSON_BINDER(ContextSpecDefaultableJsonBinder,
                               [](auto is_loading, const auto& options,
                                  auto* obj, auto* j) {
                                 return jb::DefaultInitializedValue()(
                                     is_loading, options, obj, j);
                               })

bool IsPartialBindingContext(const Context& context) {
  return internal_context::Access::impl(context)->root_->bind_partial_;
}

void SetRecordBindingState(internal::ContextSpecBuilder& builder,
                           bool record_binding_state) {
  auto& impl = internal_context::Access::impl(builder);
  auto ptr = impl.release();
  ptr.set_tag(record_binding_state);
  impl.reset(ptr, internal::adopt_object_ref);
}

}  // namespace internal

namespace internal_context {

Result<::nlohmann::json> BuilderResourceSpec::ToJson(
    Context::ToJsonOptions options) {
  ::nlohmann::json json_spec;
  if (!underlying_spec_->key_.empty()) {
    return underlying_spec_->key_;
  }
  return underlying_spec_->ToJson(options);
}

Result<ResourceImplStrongPtr> BuilderResourceSpec::CreateResource(
    const internal::ContextResourceCreationContext& creation_context) {
  return underlying_spec_->CreateResource(creation_context);
}

ResourceSpecImplPtr BuilderResourceSpec::UnbindContext(
    const internal::ContextSpecBuilder& spec_builder) {
  if (!underlying_spec_->key_.empty()) {
    return ResourceSpecImplPtr(new ResourceReference(underlying_spec_->key_));
  }
  return underlying_spec_->UnbindContext(spec_builder);
}

BuilderImpl::~BuilderImpl() {
  auto& ids = ids_;
  using SharedEntry = std::pair<ResourceImplBase*, ResourceEntry*>;
  std::vector<SharedEntry> shared_entries;
  for (auto& p : resources_) {
    const auto& key = p.first->spec_->key_;
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
        key = tensorstore::StrCat(resource->spec_->provider_->id_, "#", i);
        if (!ids.count(key)) break;
        ++i;
      }
      ids[key]++;
    }
    // Leave `entry->spec->key_` unset since `entry->spec` is exposed as a
    // resource spec accessible via a `Context::Resource` instance.
    entry->spec->underlying_spec_->key_ = key;
    root_->resources_.insert(entry->spec->underlying_spec_);
  }
}

void BuilderImplPtrTraits::increment(BuilderImplTaggedPtr p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<BuilderImpl>*>(p.get()));
}

void BuilderImplPtrTraits::decrement(BuilderImplTaggedPtr p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<BuilderImpl>*>(p.get()));
}
}  // namespace internal_context

namespace internal {
ContextSpecBuilder ContextSpecBuilder::Make(ContextSpecBuilder parent,
                                            Context::Spec existing_spec) {
  ContextSpecBuilder builder;
  if (existing_spec.impl_) {
    if (existing_spec.impl_->use_count() != 1) {
      existing_spec.impl_.reset(
          new internal_context::ContextSpecImpl(*existing_spec.impl_));
    }
  }
  if (parent.impl_) {
    builder.impl_ = std::move(parent.impl_);
    builder.spec_impl_ = std::move(existing_spec.impl_);
  } else {
    builder.impl_.reset(internal_context::BuilderImplTaggedPtr(
        new internal_context::BuilderImpl, parent.impl_.get().tag()));
    if (!existing_spec.impl_) {
      builder.spec_impl_.reset(new internal_context::ContextSpecImpl);
    } else {
      builder.spec_impl_ = std::move(existing_spec.impl_);
    }
    builder.impl_->root_ = builder.spec_impl_;
  }
  if (builder.spec_impl_ && !builder.spec_impl_->resources_.empty()) {
    auto& ids = builder.impl_->ids_;
    for (const auto& resource_spec : builder.spec_impl_->resources_) {
      ids[resource_spec->key_]++;
      // Call `UnbindContext` to ensure any resource references within
      // `resource_spec` are also marked in `ids`.  We discard the returned
      // copy, since anything in `Context::Spec` is guaranteed not to be bound.
      resource_spec->UnbindContext(builder);
    }
  }
  return builder;
}

Context::Spec ContextSpecBuilder::spec() const {
  Context::Spec spec;
  spec.impl_ = spec_impl_;
  return spec;
}
}  // namespace internal

namespace internal_context {

ResourceSpecImplPtr ResourceImplBase::UnbindContext(
    const internal::ContextSpecBuilder& spec_builder) {
  auto spec = spec_->provider_->DoGetSpec(*this, spec_builder);
  spec->provider_ = spec_->provider_;
  spec->is_default_ = spec_->is_default_;
  spec->key_ = spec_->key_;
  return spec;
}

namespace {
internal_context::ResourceSpecImplPtr AddResource(
    const internal::ContextSpecBuilder& builder,
    internal_context::ResourceImplBase* resource) {
  internal_context::ResourceImplWeakPtr resource_ptr(resource);
  auto* impl = internal_context::Access::impl(builder).get().get();
  auto& entry = impl->resources_[resource_ptr];
  if (!entry.spec) {
    // Register new resource.
    entry.spec.reset(new internal_context::BuilderResourceSpec);
    auto new_spec = entry.spec;
    entry.spec->provider_ = resource->spec_->provider_;
    entry.id = impl->next_id_++;
    entry.shared =
        (resource->spec_->is_default_ || !resource->spec_->key_.empty());
    // Note: We can't access `entry` after calling `UnbindContext`, because
    // `UnbindContext` may call back into `AddResource`, which may modify the
    // `impl->resources_` hash table and invalidate our `entry` reference.
    auto underlying_spec = resource->UnbindContext(builder);
    new_spec->underlying_spec_ = std::move(underlying_spec);
    return new_spec;
  } else {
    entry.shared = true;
    return entry.spec;
  }
}
}  // namespace

ResourceOrSpecPtr AddResourceOrSpec(const internal::ContextSpecBuilder& builder,
                                    ResourceOrSpecTaggedPtr resource_or_spec) {
  assert(internal_context::Access::impl(builder));
  if (!resource_or_spec) {
    resource_or_spec.set_tag<1>(false);
    return ResourceOrSpecPtr(resource_or_spec);
  }
  if (!IsResource(resource_or_spec)) {
    return ToResourceOrSpecPtr(
        static_cast<ResourceSpecImplBase*>(resource_or_spec.get())
            ->UnbindContext(builder));
  } else {
    auto new_ptr = ToResourceOrSpecPtr(AddResource(
        builder, static_cast<ResourceImplBase*>(resource_or_spec.get())));
    if (internal::GetRecordBindingState(builder)) {
      auto new_tagged_ptr = new_ptr.release();
      new_tagged_ptr.set_tag<1>(true);
      new_ptr = ResourceOrSpecPtr(new_tagged_ptr, internal::adopt_object_ref);
    }
    return new_ptr;
  }
}

absl::Status ResourceSpecFromJsonWithDefaults(
    std::string_view provider_id, const JsonSerializationOptions& options,
    ResourceOrSpecPtr& spec, ::nlohmann::json* j) {
  if (j->is_discarded()) {
    spec = internal_context::DefaultResourceSpec(provider_id);
  } else if (j->is_array()) {
    // Context resource marked as being bound when it was serialized.
    const auto& arr = j->get_ref<const ::nlohmann::json::array_t&>();
    if (arr.size() != 1) {
      return internal_json::ExpectedError(*j, "single-element array");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto spec_ptr, ResourceSpecFromJson(provider_id, arr[0], options));
    spec = ToResourceOrSpecPtr(std::move(spec_ptr));
    if (options.preserve_bound_context_resources_) {
      auto tagged_ptr = spec.release();
      tagged_ptr.set_tag<1>(true);
      spec = ResourceOrSpecPtr(tagged_ptr, internal::adopt_object_ref);
    }
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(auto spec_ptr,
                                 internal_context::ResourceSpecFromJson(
                                     provider_id, std::move(*j), options));
    spec = ToResourceOrSpecPtr(std::move(spec_ptr));
  }
  return absl::OkStatus();
}

absl::Status ResourceSpecToJsonWithDefaults(
    const JsonSerializationOptions& options, ResourceOrSpecTaggedPtr spec,
    ::nlohmann::json* j) {
  if (!spec || IsResource(spec)) {
    // We cannot convert a bound context resource to json.  When converting a
    // spec type like `tensorstore::kvstore::Spec` or `tensorstore::Spec`,
    // `NestedContextJsonBinder` will first unbind the context resources in
    // order to allow them to be converted to json.  However, for individual
    // `Context::Resource` objects not contained within a `Spec`-like type (not
    // very likely to occur), that is not possible since there is no
    // `Context::Spec` in which shared resources could be specified; in that
    // case, we will just skip encoding context resources.
    *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
  } else {
    auto* spec_ptr = static_cast<ResourceSpecImplBase*>(spec.get());
    TENSORSTORE_ASSIGN_OR_RETURN(*j, spec_ptr->ToJson(options));
    if (options.preserve_bound_context_resources_ &&
        IsImmediateBindingResourceSpec(spec)) {
      ::nlohmann::json::array_t arr(1);
      arr[0] = std::move(*j);
      *j = std::move(arr);
    }
    if (!IncludeDefaults(options).include_defaults() && j->is_string() &&
        j->get_ref<const std::string&>() == spec_ptr->provider_->id_) {
      *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
    }
  }
  return absl::OkStatus();
}

absl::Status GetOrCreateResource(ContextImpl* context,
                                 ResourceOrSpecTaggedPtr resource_or_spec,
                                 ResourceContainer* trigger,
                                 ResourceOrSpecPtr& resource) {
  assert(context);
  if (!resource_or_spec) {
    resource.reset();
    return absl::OkStatus();
  }
  if (IsResource(resource_or_spec)) {
    resource.reset(resource_or_spec);
    return absl::OkStatus();
  }
  if (context->root_->bind_partial_ &&
      !IsImmediateBindingResourceSpec(resource_or_spec)) {
    resource.reset(resource_or_spec);
    return absl::OkStatus();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto resource_ptr,
      internal_context::GetOrCreateResource(
          *context, static_cast<ResourceSpecImplBase&>(*resource_or_spec),
          trigger));
  resource = ToResourceOrSpecPtr(std::move(resource_ptr));
  return absl::OkStatus();
}

void StripContext(ResourceOrSpecPtr& spec) {
  if (!spec) return;
  spec = internal_context::DefaultResourceSpec(
      IsResource(spec.get())
          ? static_cast<ResourceImplBase&>(*spec).spec_->provider_->id_
          : static_cast<ResourceSpecImplBase&>(*spec).provider_->id_);
}

namespace {

/// Verifies that `provider_id` matches the provider id specified by the context
/// resource key given by `key`.
///
/// In the case of a match, returns `true`.  In the case of a mismatch, sets
/// `source` to an unealthy state and returns `false`.
[[nodiscard]] bool VerifyProviderIdMatch(serialization::DecodeSource& source,
                                         std::string_view provider_id,
                                         std::string_view key) {
  if (internal_context::ParseResourceProvider(key) == provider_id) {
    return true;
  }
  source.Fail(serialization::DecodeError(tensorstore::StrCat(
      "Context resource key ", tensorstore::QuoteString(key),
      " does not match expected provider ",
      tensorstore::QuoteString(provider_id))));
  return false;
}

/// Serializer for a context resource spec (guaranteed not to be a bound
/// resource) with a provider id that is known prior to decoding.
///
/// The resource spec is simply encoded via its JSON representation.
struct ContextResourceSpecImplSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal_context::ResourceSpecImplPtr& value,
      JsonSerializationOptions json_serialization_options = {}) {
    if (!serialization::EncodeTuple(sink, value->is_default_, value->key_)) {
      return false;
    }
    if (value->is_default_) return true;
    ::nlohmann::json json;
    TENSORSTORE_ASSIGN_OR_RETURN(
        json, value->ToJson(json_serialization_options), (sink.Fail(_), false));
    assert(!json.is_discarded());
    return serialization::Encode(sink, json);
  }
  [[nodiscard]] bool Decode(
      serialization::DecodeSource& source,
      internal_context::ResourceSpecImplPtr& value,
      JsonSerializationOptions json_serialization_options = {}) {
    bool is_default;
    std::string_view key;
    if (!serialization::DecodeTuple(source, is_default, key)) return false;
    // Warning: `key` is valid only until the next use of `source`.
    if (!key.empty() && !VerifyProviderIdMatch(source, provider_id, key)) {
      return false;
    }
    if (is_default) {
      auto& provider = internal_context::GetProviderOrDie(provider_id);
      value = MakeDefaultResourceSpec(provider, key);
    } else {
      // Make a copy of `key`, since `key` is only valid until the next use of
      // `source`.
      std::string key_copy(key);
      ::nlohmann::json json_spec;
      if (!serialization::Decode(source, json_spec)) return false;
      TENSORSTORE_ASSIGN_OR_RETURN(
          value,
          internal_context::ResourceSpecFromJson(provider_id, json_spec,
                                                 json_serialization_options),
          (source.Fail(_), false));
      value->key_ = std::move(key_copy);
    }
    return true;
  }
  std::string_view provider_id;
};

[[nodiscard]] bool EncodeContextSpecBuilder(
    serialization::EncodeSink& sink, internal::ContextSpecBuilder&& builder);

[[nodiscard]] bool DecodeContextSpecBuilder(
    serialization::DecodeSource& source,
    internal_context::ContextImplPtr& context);

/// Serializer for a context resource (guaranteed not to be an unbound resource
/// spec) with a provider id that is known prior to decoding.
struct ContextResourceImplSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal_context::ResourceImplWeakPtr& value) {
    auto creator = internal_context::GetCreator(*value);
    if (!serialization::Encode(sink, creator)) return false;
    if (creator) {
      assert(!value->spec_->key_.empty());
      return serialization::Encode(sink, value->spec_->key_);
    }
    auto builder = internal::ContextSpecBuilder::Make();
    auto spec = value->UnbindContext(builder);
    if (!internal_context::EncodeContextSpecBuilder(sink, std::move(builder))) {
      return false;
    }
    return ContextResourceSpecImplSerializer::Encode(sink, spec);
  }

  [[nodiscard]] bool Decode(
      serialization::DecodeSource& source,
      internal_context::ResourceImplWeakPtr& value) const {
    internal_context::ContextImplPtr creator;
    if (!serialization::Decode(source, creator)) return false;
    if (creator) {
      return DecodeByReferenceToExistingContext(source, *creator, value);
    }

    // Decode by spec.

    internal_context::ContextImplPtr context_impl;
    if (!internal_context::DecodeContextSpecBuilder(source, context_impl)) {
      return false;
    }
    internal_context::ResourceSpecImplPtr resource_spec;
    if (!ContextResourceSpecImplSerializer{provider_id}.Decode(source,
                                                               resource_spec)) {
      return false;
    }
    // Don't set key until after getting the resource, since
    // `GetOrCreateResource` expects any `resource_spec` with a key to be
    // internal to the same `context_impl`.
    std::string key;
    std::swap(key, resource_spec->key_);
    TENSORSTORE_ASSIGN_OR_RETURN(value,
                                 internal_context::GetOrCreateResource(
                                     *context_impl, *resource_spec, nullptr),
                                 (source.Fail(_), false));
    resource_spec->key_ = std::move(key);
    return true;
  }

  [[nodiscard]] bool DecodeByReferenceToExistingContext(
      serialization::DecodeSource& source,
      internal_context::ContextImpl& creator,
      internal_context::ResourceImplWeakPtr& value) const {
    std::string_view key;
    if (!serialization::Decode(source, key)) return false;
    if (!VerifyProviderIdMatch(source, provider_id, key)) return false;
    // Warning: `key` is valid only until the next use of `source`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto spec, internal_context::ResourceSpecFromJson(provider_id, key, {}),
        (source.Fail(_), false));
    TENSORSTORE_ASSIGN_OR_RETURN(
        value, internal_context::GetOrCreateResource(creator, *spec, nullptr),
        (source.Fail(_), false));
    return true;
  }

  std::string_view provider_id;
};

/// Encodes resources added to a `ContextSpecBuilder`.
///
/// This is used to encode the resources transitively required by another
/// resource.
///
/// Resources are encoded indirectly, meaning a given resource is only encoded
/// once even if referenced multiple times.
///
/// \param sink Encode sink to use.
/// \param builder Context spec builder to encode, must be the last remaining
///     reference.
bool EncodeContextSpecBuilder(serialization::EncodeSink& sink,
                              internal::ContextSpecBuilder&& builder) {
  std::vector<
      std::pair<internal_context::ResourceImplWeakPtr,
                internal::IntrusivePtr<internal_context::BuilderResourceSpec>>>
      deps;
  auto& resources = internal_context::Access::impl(builder)->resources_;
  deps.reserve(resources.size());
  // Make a copy of the resources before destroying `builder`.
  for (auto& [resource, entry] : resources) {
    deps.emplace_back(resource, entry.spec);
    entry.shared = true;
  }
  ABSL_CHECK_EQ(internal_context::Access::impl(builder)->use_count(), 1);
  // Rely on builder's destructor to update all of the spec keys in
  // `deps`.
  builder = internal::ContextSpecBuilder();
  if (!serialization::WriteSize(sink.writer(), deps.size())) return false;
  for (size_t i = 0; i < deps.size(); ++i) {
    auto& [dep_resource, dep_spec] = deps[i];
    if (!serialization::Encode(sink, dep_spec->underlying_spec_->key_)) {
      return false;
    }
    if (!sink.Indirect(dep_resource, ContextResourceImplSerializer{})) {
      return false;
    }
  }
  return true;
}

/// Decodes a single context resource that was encoded by
/// `EncodeContextSpecBuilder`.
[[nodiscard]] bool DecodeContextResourceInContextSpecBuilder(
    serialization::DecodeSource& source,
    internal_context::ContextImpl& context_impl) {
  std::string key;
  if (!serialization::Decode(source, key)) return false;
  internal_context::ResourceImplWeakPtr resource;
  std::string_view provider_id = internal_context::ParseResourceProvider(key);
  if (!source.Indirect(resource, ContextResourceImplSerializer{provider_id})) {
    return false;
  }
  if (resource->spec_->provider_->id_ != provider_id) {
    source.Fail(serialization::DecodeError(tensorstore::StrCat(
        "Context resource has provider id ",
        tensorstore::QuoteString(resource->spec_->provider_->id_),
        " but expected ", tensorstore::QuoteString(provider_id))));
    return false;
  }
  auto container = std::make_unique<internal_context::ResourceContainer>();
  if (resource->spec_->key_ != key) {
    // Wrap the spec in a `BuilderResourceSpec` in order to allow it to be
    // stored under a different key.
    container->spec_.reset(new internal_context::BuilderResourceSpec);
    container->spec_->provider_ = resource->spec_->provider_;
    container->spec_->key_ = std::move(key);
    static_cast<internal_context::BuilderResourceSpec&>(*container->spec_)
        .underlying_spec_ = resource->spec_;
  } else {
    container->spec_ = resource->spec_;
  }
  container->result_.emplace(resource.get());
  if (!context_impl.spec_->resources_.emplace(container->spec_).second) {
    // Keys are not unique.
    source.Fail(absl::DataLossError(
        tensorstore::StrCat("Duplicate context resource key in Context spec ",
                            tensorstore::QuoteString(container->spec_->key_))));
    return false;
  }
  [[maybe_unused]] bool inserted_resource =
      context_impl.resources_.emplace(std::move(container)).second;
  // Insertion that can't fail since we already determined the spec key is
  // unique.
  assert(inserted_resource);
  return true;
}

/// Decodes context resources encoded by `EncodeContextSpecBuilder`.
bool DecodeContextSpecBuilder(serialization::DecodeSource& source,
                              internal_context::ContextImplPtr& context) {
  size_t count;
  if (!serialization::ReadSize(source.reader(), count)) return false;
  internal_context::ContextImplPtr context_impl(
      new internal_context::ContextImpl);
  context_impl->spec_.reset(new internal_context::ContextSpecImpl);
  context_impl->root_ = context_impl.get();
  while (count--) {
    if (!DecodeContextResourceInContextSpecBuilder(source, *context_impl)) {
      return false;
    }
  }
  context = std::move(context_impl);
  return true;
}

[[nodiscard]] bool EncodeContextResource(
    serialization::EncodeSink& sink,
    const internal_context::ResourceImplWeakPtr& resource) {
  return serialization::IndirectPointerSerializer<
             internal_context::ResourceImplWeakPtr,
             ContextResourceImplSerializer>()
      .Encode(sink, resource);
}

[[nodiscard]] bool DecodeContextResource(
    serialization::DecodeSource& source, std::string_view provider_id,
    internal_context::ResourceImplWeakPtr& resource) {
  return serialization::IndirectPointerSerializer<
             internal_context::ResourceImplWeakPtr,
             ContextResourceImplSerializer>{{provider_id}}
      .Decode(source, resource);
}

}  // namespace

bool EncodeContextResourceOrSpec(
    serialization::EncodeSink& sink,
    const internal_context::ResourceOrSpecPtr& resource) {
  const bool is_resource = internal_context::IsResource(resource.get());
  if (!serialization::Encode(sink, is_resource)) return false;
  if (is_resource) {
    return EncodeContextResource(
        sink, internal_context::ResourceImplWeakPtr(
                  static_cast<internal_context::ResourceImplBase*>(
                      resource.get().get())));
  } else {
    return ContextResourceSpecImplSerializer::Encode(
        sink, internal_context::ResourceSpecImplPtr(
                  static_cast<internal_context::ResourceSpecImplBase*>(
                      resource.get().get())));
  }
}

bool DecodeContextResourceOrSpec(
    serialization::DecodeSource& source, std::string_view provider_id,
    internal_context::ResourceOrSpecPtr& resource) {
  bool is_resource;
  if (!serialization::Decode(source, is_resource)) return false;
  if (is_resource) {
    internal_context::ResourceImplWeakPtr resource_ptr;
    if (!DecodeContextResource(source, provider_id, resource_ptr)) return false;
    resource = internal_context::ToResourceOrSpecPtr(std::move(resource_ptr));
  } else {
    internal_context::ResourceSpecImplPtr spec_ptr;
    if (!ContextResourceSpecImplSerializer{provider_id}.Decode(source,
                                                               spec_ptr)) {
      return false;
    }
    resource = internal_context::ToResourceOrSpecPtr(std::move(spec_ptr));
  }
  return true;
}

bool ContextSpecImplPtrNonNullDirectSerializer::Encode(
    serialization::EncodeSink& sink,
    const internal_context::ContextSpecImplPtr& value) {
  Context::Spec spec;
  internal_context::Access::impl(spec) = value;
  return serialization::JsonBindableSerializer<Context::Spec>::Encode(sink,
                                                                      spec);
}

bool ContextSpecImplPtrNonNullDirectSerializer::Decode(
    serialization::DecodeSource& source,
    internal_context::ContextSpecImplPtr& value) {
  Context::Spec spec;
  if (!serialization::JsonBindableSerializer<Context::Spec>::Decode(source,
                                                                    spec)) {
    return false;
  }
  value = internal_context::Access::impl(spec);
  return true;
}

bool ContextImplPtrNonNullDirectSerializer::Encode(
    serialization::EncodeSink& sink,
    const internal_context::ContextImplPtr& value) {
  return serialization::EncodeTuple(sink, value->spec_, value->parent_);
}

bool ContextImplPtrNonNullDirectSerializer::Decode(
    serialization::DecodeSource& source,
    internal_context::ContextImplPtr& value) {
  Context::Spec spec;
  Context parent;
  if (!serialization::DecodeTuple(source, spec, parent)) return false;
  Context context(std::move(spec), std::move(parent));
  value = std::move(internal_context::Access::impl(context));
  return true;
}

bool UntypedContextResourceImplPtrNonNullDirectSerializer::Encode(
    serialization::EncodeSink& sink,
    const internal_context::ResourceImplWeakPtr& value) {
  std::string_view provider_id = value->spec_->provider_->id_;
  if (!serialization::Encode(sink, provider_id)) return false;
  return ContextResourceImplSerializer{provider_id}.Encode(sink, value);
}

bool UntypedContextResourceImplPtrNonNullDirectSerializer::Decode(
    serialization::DecodeSource& source,
    internal_context::ResourceImplWeakPtr& value) {
  std::string provider_id;
  if (!serialization::Decode(source, provider_id)) return false;
  if (!internal_context::GetProvider(provider_id)) {
    source.Fail(internal_context::ProviderNotRegisteredError(provider_id));
    return false;
  }
  return ContextResourceImplSerializer{provider_id}.Decode(source, value);
}

}  // namespace internal_context

namespace serialization {

bool Serializer<Context::Spec>::Encode(EncodeSink& sink,
                                       const Context::Spec& value) {
  return serialization::Encode(sink, internal_context::Access::impl(value));
}

bool Serializer<Context::Spec>::Decode(DecodeSource& source,
                                       Context::Spec& value) {
  return serialization::Decode(source, internal_context::Access::impl(value));
}

bool Serializer<Context>::Encode(EncodeSink& sink, const Context& value) {
  return serialization::Encode(sink, internal_context::Access::impl(value));
}

bool Serializer<Context>::Decode(DecodeSource& source, Context& value) {
  return serialization::Decode(source, internal_context::Access::impl(value));
}

}  // namespace serialization
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_context::ContextSpecImplPtr,
    (tensorstore::serialization::IndirectPointerSerializer<
        tensorstore::internal_context::ContextSpecImplPtr,
        tensorstore::internal_context::
            ContextSpecImplPtrNonNullDirectSerializer>{}))

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_context::ContextImplPtr,
    (tensorstore::serialization::IndirectPointerSerializer<
        tensorstore::internal_context::ContextImplPtr,
        tensorstore::internal_context::
            ContextImplPtrNonNullDirectSerializer>{}))
