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

#ifndef TENSORSTORE_INTERNAL_CACHE_KEY_ABSL_TIME_H_
#define TENSORSTORE_INTERNAL_CACHE_KEY_ABSL_TIME_H_

#include "absl/time/time.h"
#include "tensorstore/internal/cache_key/cache_key.h"

namespace tensorstore {
namespace internal {

template <>
struct CacheKeyEncoder<absl::Duration> {
  static void Encode(std::string* out, const absl::Duration& d) {
    if (d == absl::InfiniteDuration()) {
      EncodeCacheKey(out, 0);
    } else {
      EncodeCacheKey(out, 1, absl::ToInt64Nanoseconds(d));
    }
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KEY_ABSL_TIME_H_
