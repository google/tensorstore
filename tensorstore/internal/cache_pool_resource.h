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

#ifndef TENSORSTORE_INTERNAL_CACHE_POOL_RESOURCE_H_
#define TENSORSTORE_INTERNAL_CACHE_POOL_RESOURCE_H_

#include "tensorstore/internal/cache.h"

namespace tensorstore {
namespace internal {

/// Context resource corresponding to a CachePool.
struct CachePoolResource {
  static constexpr char id[] = "cache_pool";
  using Resource = CachePool::WeakPtr;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_POOL_RESOURCE_H_
