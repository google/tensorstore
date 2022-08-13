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

#include "tensorstore/internal/meta.h"

#include <type_traits>

namespace {

using ::tensorstore::internal::GetFirstArgument;

static_assert(
    std::is_same_v<int&, decltype(GetFirstArgument(std::declval<int&>(),
                                                   std::declval<float&>()))>);

static_assert(std::is_same_v<
              const int&, decltype(GetFirstArgument(std::declval<const int&>(),
                                                    std::declval<float&>()))>);

static_assert(
    std::is_same_v<int&&, decltype(GetFirstArgument(std::declval<int>(),
                                                    std::declval<float&>()))>);

static_assert(GetFirstArgument(3, 4) == 3);

}  // namespace
