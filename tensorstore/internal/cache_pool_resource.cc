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

#include "tensorstore/internal/cache_pool_resource.h"

#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {
namespace {

struct CachePoolResourceTraits
    : public ContextResourceTraits<CachePoolResource> {
  using Spec = CachePool::Limits;
  using Resource = typename CachePoolResource::Resource;
  static constexpr Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal::json_binding;
    return jb::Object(
        jb::Member("total_bytes_limit",
                   jb::Projection(&Spec::total_bytes_limit,
                                  jb::DefaultValue([](auto* v) { *v = 0; }))),
        jb::Member(
            "queued_for_writeback_bytes_limit",
            jb::Dependent(
                [](auto is_loading, const auto& options, auto* obj, auto* j) {
                  return jb::Projection(
                      &Spec::queued_for_writeback_bytes_limit,
                      jb::DefaultValue(
                          [obj](auto* v) { *v = obj->total_bytes_limit / 2; },
                          jb::Integer<std::size_t>(0, obj->total_bytes_limit)));
                })));
  }
  static Result<Resource> Create(const Spec& limits,
                                 ContextResourceCreationContext context) {
    return CachePool::WeakPtr(CachePool::Make(limits));
  }

  static Spec GetSpec(const Resource& pool, const ContextSpecBuilder& builder) {
    return pool->limits();
  }
  static void AcquireStrongReference(const Resource* p) {
    internal_cache::StrongPtrTraitsCachePool::increment(p->get());
  }
  static void ReleaseStrongReference(const Resource* p) {
    internal_cache::StrongPtrTraitsCachePool::decrement(p->get());
  }
};

const ContextResourceRegistration<CachePoolResourceTraits> registration;

}  // namespace
}  // namespace internal
}  // namespace tensorstore
