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

#ifndef TENSORSTORE_INTERNAL_CACHE_POOL_LIMITS_H_
#define TENSORSTORE_INTERNAL_CACHE_POOL_LIMITS_H_

#include <cstddef>

namespace tensorstore {
namespace internal {

/// Memory limit parameters for a cache pool.
struct CachePoolLimits {
  std::size_t total_bytes_limit = 0;
  std::size_t queued_for_writeback_bytes_limit = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_POOL_LIMITS_H_
