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

#ifndef TENSORSTORE_UTIL_APPLY_MEMBERS_STD_COMPLEX_H_
#define TENSORSTORE_UTIL_APPLY_MEMBERS_STD_COMPLEX_H_

#include <complex>
#include <type_traits>

#include "tensorstore/util/apply_members/apply_members.h"

namespace tensorstore {

template <typename T>
struct ApplyMembers<std::complex<T>> {
  template <typename F>
  static constexpr auto Apply(std::complex<T>& x, F f) {
    // C++11 allows this reinterpret_cast to a 2-element array of the base type
    // as a special case for C compatibility.
    auto& arr = reinterpret_cast<T(&)[2]>(x);
    return f(arr[0], arr[1]);
  }

  template <typename F>
  static constexpr auto Apply(const std::complex<T>& x, F f) {
    auto& arr = reinterpret_cast<const T(&)[2]>(x);
    return f(arr[0], arr[1]);
  }
};

template <typename T>
constexpr inline bool SerializeUsingMemcpy<std::complex<T>> =
    SerializeUsingMemcpy<T>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_APPLY_MEMBERS_STD_COMPLEX_H_
