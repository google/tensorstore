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

#ifndef TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_VARIANT_H_
#define TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_VARIANT_H_

#include <type_traits>
#include <variant>

#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace garbage_collection {

template <typename... T>
struct GarbageCollection<
    std::variant<T...>,
    std::enable_if_t<!std::is_trivially_destructible_v<std::variant<T...>>>> {
  constexpr static bool required() {
    return (IsGarbageCollectionRequired<T> || ...);
  }

  static void Visit(GarbageCollectionVisitor& visitor,
                    const std::variant<T...>& value) {
    if (value.valueless_by_exception()) return;
    std::visit(
        [&visitor](const auto& element) {
          garbage_collection::GarbageCollectionVisit(visitor, element);
        },
        value);
  }
};

}  // namespace garbage_collection
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_GARBAGE_COLLECTION_STD_VARIANT_H_
