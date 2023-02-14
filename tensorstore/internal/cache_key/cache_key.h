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

#ifndef TENSORSTORE_INTERNAL_CACHE_KEY_CACHE_KEY_H_
#define TENSORSTORE_INTERNAL_CACHE_KEY_CACHE_KEY_H_

#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "tensorstore/internal/cache_key/fwd.h"
#include "tensorstore/util/apply_members/apply_members.h"

namespace tensorstore {
namespace internal {

/// Wrapper for use with `ApplyMembers` that indicates a value that should not
/// be included in the cache key.
template <typename T>
struct CacheKeyExcludes {
  T value;

  /// When not used with `EncodeCacheKey`, `CacheKeyExcludes` just passes
  /// through the wrapped value.
  template <typename X, typename F>
  static constexpr auto ApplyMembers(X&& x, F f) {
    return f(x.value);
  }
};

template <typename T>
CacheKeyExcludes(T&& x) -> CacheKeyExcludes<T>;

template <typename... U>
void EncodeCacheKey(std::string* out, const U&... u);

// Unused, but allows other `EncodeCacheKeyAdl` methods to be found via
// argument-dependent lookup (ADL).
inline void EncodeCacheKeyAdl() {}

template <typename T, typename SFINAE>
struct CacheKeyEncoder {
  static void Encode(std::string* out, const T& value) {
    EncodeCacheKeyAdl(out, value);
  }
};

template <typename T>
struct CacheKeyEncoder<T, std::enable_if_t<SerializeUsingMemcpy<T>>> {
  static void Encode(std::string* out, T value) {
    out->append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
};

template <>
struct CacheKeyEncoder<std::string_view> {
  static void Encode(std::string* out, std::string_view k) {
    EncodeCacheKey(out, k.size());
    out->append(k.data(), k.size());
  }
};

template <>
struct CacheKeyEncoder<std::string> : public CacheKeyEncoder<std::string_view> {
};

template <>
struct CacheKeyEncoder<std::type_info> {
  static void Encode(std::string* out, const std::type_info& t) {
    EncodeCacheKey(out, std::string_view(t.name()));
  }
};

template <typename T>
struct CacheKeyEncoder<CacheKeyExcludes<T>> {
  static void Encode(std::string* out, const CacheKeyExcludes<T>& v) {
    // do nothing
  }
};

template <typename T>
constexpr inline bool IsCacheKeyExcludes = false;

template <typename T>
constexpr inline bool IsCacheKeyExcludes<CacheKeyExcludes<T>> = true;

template <typename T>
struct CacheKeyEncoder<
    T, std::enable_if_t<SupportsApplyMembers<T> && !IsCacheKeyExcludes<T> &&
                        !SerializeUsingMemcpy<T>>> {
  static void Encode(std::string* out, const T& v) {
    ApplyMembers<T>::Apply(
        v, [&out](auto&&... x) { (internal::EncodeCacheKey(out, x), ...); });
  }
};

template <typename... U>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void EncodeCacheKey(std::string* out,
                                                        const U&... u) {
  (CacheKeyEncoder<U>::Encode(out, u), ...);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KEY_CACHE_KEY_H_
