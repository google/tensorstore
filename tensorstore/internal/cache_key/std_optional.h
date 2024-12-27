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

#ifndef TENSORSTORE_INTERNAL_CACHE_KEY_STD_OPTIONAL_H_
#define TENSORSTORE_INTERNAL_CACHE_KEY_STD_OPTIONAL_H_

#include <optional>
#include <string>

#include "tensorstore/internal/cache_key/cache_key.h"

namespace tensorstore {
namespace internal {

template <typename T>
struct CacheKeyEncoder<std::optional<T>> {
  static void Encode(std::string* out, const std::optional<T>& v) {
    internal::EncodeCacheKey(out, v.has_value());
    if (v) internal::EncodeCacheKey(out, *v);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KEY_STD_OPTIONAL_H_
