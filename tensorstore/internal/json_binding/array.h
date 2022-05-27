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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_ARRAY_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_ARRAY_H_

/// \file
/// Functions for converting between `::nlohmann::json` and
/// `tensorstore::Array`.

#include <assert.h>

#include <type_traits>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_json_binding {

/// Returns a binder for a nested JSON array.
constexpr auto NestedVoidArray(DataType dtype,
                               DimensionIndex rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    using Element = typename internal::remove_cvref_t<
        std::remove_pointer_t<decltype(obj)>>::Element;
    static_assert(std::is_void_v<Element>,
                  "Use NestedArray for tensorstore::Array<T> arrays");
    if constexpr (is_loading) {
      TENSORSTORE_ASSIGN_OR_RETURN(*obj, internal_json::JsonParseNestedArray(
                                             *j, dtype, rank_constraint));
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(*j,
                                   internal_json::JsonEncodeNestedArray(*obj));
    }
    return absl::OkStatus();
  };
}

/// Returns a binder for a nested JSON array.
constexpr auto NestedArray(DimensionIndex rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    using Element = typename internal::remove_cvref_t<
        std::remove_pointer_t<decltype(obj)>>::Element;
    static_assert(!std::is_void_v<Element>,
                  "Use NestedVoidArray for tensorstore::Array<void> arrays");
    if constexpr (is_loading) {
      TENSORSTORE_ASSIGN_OR_RETURN(SharedArray<void> array,
                                   internal_json::JsonParseNestedArray(
                                       *j, dtype_v<Element>, rank_constraint));
      *obj = StaticDataTypeCast<Element, unchecked>(array);
    } else {
      if (obj->data() != nullptr) {
        *j = internal_json::JsonEncodeNestedArray(
            *obj, [](const void* value) -> ::nlohmann::json {
              assert(value);
              return *reinterpret_cast<const Element*>(value);
            });
      }
    }
    return absl::OkStatus();
  };
}

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_ARRAY_H_
