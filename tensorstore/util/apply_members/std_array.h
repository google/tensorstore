// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_UTIL_APPLY_MEMBERS_STD_ARRAY_H_
#define TENSORSTORE_UTIL_APPLY_MEMBERS_STD_ARRAY_H_

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "tensorstore/util/apply_members/apply_members.h"

namespace tensorstore {

template <typename T, size_t N>
struct ApplyMembers<T[N]> {
  template <typename X, typename F>
  static constexpr auto Apply(X&& x, F f) {
    return ApplyImpl(x, f, std::make_index_sequence<N>{});
  }
  template <typename X, typename F, size_t... Is>
  static constexpr auto ApplyImpl(X&& x, F f, std::index_sequence<Is...>) {
    return f(x[Is]...);
  }
};

template <typename T, size_t N>
struct ApplyMembers<std::array<T, N>> : public ApplyMembers<T[N]> {};

template <typename T, size_t N>
constexpr static bool SerializeUsingMemcpy<T[N]> = SerializeUsingMemcpy<T>;

template <typename T, size_t N>
constexpr static bool SerializeUsingMemcpy<std::array<T, N>> =
    SerializeUsingMemcpy<T>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_APPLY_MEMBERS_STD_ARRAY_H_
