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

#ifndef TENSORSTORE_INTERNAL_CACHE_ASYNC_INITIALIZED_CACHE_MIXIN_H_
#define TENSORSTORE_INTERNAL_CACHE_ASYNC_INITIALIZED_CACHE_MIXIN_H_

#include <string_view>
#include <type_traits>

#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

/// Mixin for classes that inherit from `tensorstore::internal::Cache` that
/// simplifies asynchronous initialization.
class AsyncInitializedCacheMixin {
 public:
  /// Future that becomes ready when the asynchronous initialization completes.
  Future<const void> initialized_;
};

/// Get or create a cache for the specified identifier.
///
/// \tparam CacheType The derived cache type, must inherit from `Cache` and
///     `AsyncInitializedCacheMixin`.
/// \param pool The cache pool in which to create the cache.
/// \param cache_identifier String that, along with the `CacheType`, identifies
///     the cache within the `pool`.
/// \param make_cache Callable compatible with signature
///     `std::unique_ptr<Cache> ()` called to create the cache.  This should
///     perform any synchronous initialization required before other callers can
///     safely access the cache.  No locks are held while this function is
///     called, and in the event of multiple concurrent calls with the same
///     `CacheType`, `cache_identifier`, and `pool`, this function may be called
///     multiple times.
/// \param async_initialize Callable compatible with signature
///     `void (Promise<void> promise, CachePtr<CacheType> cache)`, called
///     exactly once to perform any asynchronous initialization.  The `promise`
///     corresponds to `cache->initialized_`.
/// \returns The existing or new cache.
template <typename CacheType, typename MakeCache, typename AsyncInitialize>
CachePtr<CacheType> GetOrCreateAsyncInitializedCache(
    CachePool& pool, std::string_view cache_identifier, MakeCache make_cache,
    AsyncInitialize async_initialize) {
  static_assert(std::is_base_of_v<AsyncInitializedCacheMixin, CacheType>,
                "CacheType must inherit from AsyncInitializedCacheMixin");
  Promise<void> initialized_promise;
  CacheType* created_cache = nullptr;
  auto cache = pool.GetCache<CacheType>(cache_identifier, [&] {
    auto cache = make_cache();
    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    cache->initialized_ = std::move(future);
    initialized_promise = std::move(promise);
    created_cache = cache.get();
    return cache;
  });
  // Even if we just created a new cache, it is possible that another cache
  // for the same cache_identifier was created concurrently, in which case
  // the cache we just created should be discarded.
  if (created_cache && cache.get() == created_cache) {
    async_initialize(initialized_promise, cache);
  }
  return cache;
}

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_CACHE_ASYNC_INITIALIZED_CACHE_MIXIN_H_
