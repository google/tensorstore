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
    // TODO(jbms): replace with FunctionView type
    const std::function<::nlohmann::json(const void*)>& encode_element);
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
/// \param data_type Specifies the element type of the array.  Must be valid.
/// \param decode_element Function that decodes a single JSON.  On success, it
///     should decode the JSON value `v` and stores the result in `out`, which
///     is a non-null output pointer to a value of the type corresponding to
///     `data_type`.  On failure, it should return an error `Status` value.
/// \returns The parsed array on success, or the first error returned by
///     `decode_element`.
/// \dchecks `data_type.valid()`
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a nested list with
///     uniform shape.
Result<SharedArray<void>> JsonParseNestedArray(
    const ::nlohmann::json& j_root, DataType data_type,
    // TODO(jbms): replace with FunctionView type
    const std::function<Status(const ::nlohmann::json& v, void* out)>&
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
                      j, DataTypeOf<Element>(),
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
    const ::nlohmann::json& j, DataType data_type,
    DimensionIndex rank_constraint = dynamic_rank);

namespace json_binding {

/// Returns a binder for a nested JSON array.
constexpr auto NestedArray(DataType data_type,
                           DimensionIndex rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> Status {
    if constexpr (is_loading) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          *obj, internal::JsonParseNestedArray(*j, data_type, rank_constraint));
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(*j, internal::JsonEncodeNestedArray(*obj));
    }
    return absl::OkStatus();
  };
}

}  // namespace json_binding

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_ARRAY_H_
