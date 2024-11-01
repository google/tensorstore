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

#include "tensorstore/internal/cache/cache_pool_resource.h"

#include <optional>
#include <utility>

#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {
namespace {

struct CachePoolResourceTraits
    : public ContextResourceTraits<CachePoolResource> {
  // Specifies cache pool limits, or `nullopt` to indicate the cache is disabled
  // completely.
  using Spec = std::optional<CachePool::Limits>;
  using Resource = typename CachePoolResource::Resource;
  static constexpr Spec Default() { return Spec{std::in_place}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member(
            "disabled",
            jb::GetterSetter(
                [](auto& obj) { return !obj.has_value(); },
                [](auto& obj, bool disabled) {
                  if (disabled) {
                    obj = std::nullopt;
                  } else {
                    obj.emplace();
                  }
                },
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())),
        [](auto is_loading, const auto& options, auto* obj, auto* j) {
          if (!*obj) return absl::OkStatus();
          return jb::Member(
              "total_bytes_limit",
              jb::Projection(&CachePool::Limits::total_bytes_limit,
                             jb::DefaultValue([](auto* v) { *v = 0; })))(
              is_loading, options, obj, j);
        });
  }
  static Result<Resource> Create(const Spec& limits,
                                 ContextResourceCreationContext context) {
    if (!limits) return CachePool::WeakPtr();
    return CachePool::WeakPtr(CachePool::Make(*limits));
  }

  static Spec GetSpec(const Resource& pool, const ContextSpecBuilder& builder) {
    return pool ? Spec(pool->limits()) : Spec(std::nullopt);
  }
  static void AcquireContextReference(const Resource& p) {
    if (p) {
      internal_cache::StrongPtrTraitsCachePool::increment(p.get());
    }
  }
  static void ReleaseContextReference(const Resource& p) {
    if (p) {
      internal_cache::StrongPtrTraitsCachePool::decrement(p.get());
    }
  }
};

const ContextResourceRegistration<CachePoolResourceTraits> registration;

}  // namespace
}  // namespace internal
}  // namespace tensorstore
