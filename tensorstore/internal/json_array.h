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

#ifndef TENSORSTORE_INTERNAL_JSON_ARRAY_H_
#define TENSORSTORE_INTERNAL_JSON_ARRAY_H_

/// \file
/// Functions for converting between `::nlohmann::json` and
/// `tensorstore::Array`.

#include <functional>
#include <type_traits>

#include "absl/functional/function_ref.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_json {
::nlohmann::json JsonEncodeNestedArray(
    ArrayView<const void, dynamic_rank, offset_origin> array,
    absl::FunctionRef<::nlohmann::json(const void*)> encode_element);

}  // namespace internal_json

namespace internal {

/// Encodes a multi-dimensional array as a nested JSON array.
/// \tparam A An instance of `Array`.
/// \tparam F A function object with a signature compatible with
///     `::nlohmann::json (const A::Element*)`.
/// \param array The array to encode.
/// \param encode_element Function called to encode each element of the array.
/// \returns The nested array JSON representation.
template <typename A, typename F>
std::enable_if_t<IsArray<A>::value, ::nlohmann::json> JsonEncodeNestedArray(
    const A& array, F encode_element) {
  return internal_json::JsonEncodeNestedArray(
      array, [encode_element](const void* value) {
        return encode_element(static_cast<const typename A::Element*>(value));
      });
}

/// Same as above, but uses `DataType`-defined conversion to json.
Result<::nlohmann::json> JsonEncodeNestedArray(ArrayView<const void> array);

/// Parses a multi-dimensional array from a nested JSON array.
/// \param j The JSON value to parse.
/// \param dtype Specifies the element type of the array.  Must be valid.
/// \param decode_element Function that decodes a single JSON.  On success, it
///     should decode the JSON value `v` and stores the result in `out`, which
///     is a non-null output pointer to a value of the type corresponding to
///     `dtype`.  On failure, it should return an error `Status` value.
/// \returns The parsed array on success, or the first error returned by
///     `decode_element`.
/// \dchecks `dtype.valid()`
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a nested list with
///     uniform shape.
Result<SharedArray<void>> JsonParseNestedArray(
    const ::nlohmann::json& j_root, DataType dtype,
    absl::FunctionRef<Status(const ::nlohmann::json& v, void* out)>
        decode_element);

/// Parses a multi-dimensional array from a nested JSON array.
///
/// This is a convenience interface to the type-erased variant defined above.
///
/// \param j The JSON value to parse.
/// \param decode_element A function object with a signature
///     `Result<Element> (const ::nlohmann::json& v)` that decodes a single JSON
///     value.
/// \returns The parsed array on success, or the first error returned by
///     `decode_element`.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a nested list with
///     uniform shape.
template <typename DecodeElement>
Result<SharedArray<UnwrapResultType<
    std::invoke_result_t<DecodeElement, const ::nlohmann::json&>>>>
JsonParseNestedArray(const ::nlohmann::json& j, DecodeElement decode_element) {
  using Element = UnwrapResultType<
      std::invoke_result_t<DecodeElement, const ::nlohmann::json&>>;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array, JsonParseNestedArray(
                      j, dtype_v<Element>,
                      [decode_element](const ::nlohmann::json& v, void* out) {
                        TENSORSTORE_ASSIGN_OR_RETURN(
                            *static_cast<Element*>(out), decode_element(v));
                        return absl::OkStatus();
                      }));
  return StaticDataTypeCast<Element, unchecked>(array);
}

/// Same as `JsonParseNestedArray` above, but uses `DataType`-defined conversion
/// from json.
Result<SharedArray<void>> JsonParseNestedArray(
    const ::nlohmann::json& j, DataType dtype,
    DimensionIndex rank_constraint = dynamic_rank);

}  // namespace internal
namespace internal_json_binding {

/// Returns a binder for a nested JSON array.
constexpr auto NestedVoidArray(DataType dtype,
                               DimensionIndex rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    using Element = typename internal::remove_cvref_t<
        std::remove_pointer_t<decltype(obj)>>::Element;
    static_assert(std::is_void_v<Element>,
                  "Use NestedArray for tensorstore::Array<T> arrays");
    if constexpr (is_loading) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          *obj, internal::JsonParseNestedArray(*j, dtype, rank_constraint));
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(*j, internal::JsonEncodeNestedArray(*obj));
    }
    return absl::OkStatus();
  };
}

/// Returns a binder for a nested JSON array.
constexpr auto NestedArray(DimensionIndex rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    using Element = typename internal::remove_cvref_t<
        std::remove_pointer_t<decltype(obj)>>::Element;
    static_assert(!std::is_void_v<Element>,
                  "Use NestedVoidArray for tensorstore::Array<void> arrays");
    if constexpr (is_loading) {
      TENSORSTORE_ASSIGN_OR_RETURN(SharedArray<void> array,
                                   internal::JsonParseNestedArray(
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

#endif  // TENSORSTORE_INTERNAL_JSON_ARRAY_H_
