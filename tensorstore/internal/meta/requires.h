// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_META_REQUIRES_H_
#define TENSORSTORE_INTERNAL_META_REQUIRES_H_

#include <type_traits>

namespace tensorstore {
namespace internal_meta {

// Requires is a sfinae helper that detects callability.
// Example:
//  if constexpr (Requires<T>([](auto&& v) -> decltype(v.begin()){})) {
//     t.begin();
//  }
template <typename... T, typename F>
constexpr bool Requires(F) {
  return std::is_invocable_v<F, T...>;
}

}  // namespace internal_meta
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_META_REQUIRES_H_
