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

#ifndef TENSORSTORE_UTIL_APPLY_MEMBERS_STD_TUPLE_H_
#define TENSORSTORE_UTIL_APPLY_MEMBERS_STD_TUPLE_H_

#include <tuple>
#include <type_traits>

#include "tensorstore/util/apply_members/apply_members.h"

namespace tensorstore {

template <typename... T>
struct ApplyMembers<std::tuple<T...>,
                    std::enable_if_t<!std::is_empty_v<std::tuple<T...>>>> {
  template <typename X, typename F>
  static constexpr auto Apply(X&& x, F f) {
    return std::apply(f, x);
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_APPLY_MEMBERS_STD_TUPLE_H_
