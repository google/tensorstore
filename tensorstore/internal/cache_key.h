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

#ifndef TENSORSTORE_INTERNAL_CACHE_KEY_H_
#define TENSORSTORE_INTERNAL_CACHE_KEY_H_

#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "absl/strings/string_view.h"

namespace tensorstore {
namespace internal {

template <typename... U>
void EncodeCacheKey(std::string* out, const U&... u);

template <typename T>
inline std::enable_if_t<std::is_arithmetic_v<T> || std::is_enum_v<T>>
EncodeCacheKeyAdl(std::string* out, T value) {
  out->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void EncodeCacheKeyAdl(std::string* out, absl::string_view k) {
  EncodeCacheKey(out, k.size());
  out->append(k.data(), k.size());
}

inline void EncodeCacheKeyAdl(std::string* out, const std::type_info& t) {
  EncodeCacheKey(out, t.name());
}

template <typename T>
void EncodeCacheKeyAdl(std::string* out, const std::optional<T>& v) {
  EncodeCacheKey(out, v.has_value());
  if (v) EncodeCacheKey(out, *v);
}

template <typename... U>
void EncodeCacheKey(std::string* out, const U&... u) {
  (EncodeCacheKeyAdl(out, u), ...);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KEY_H_
