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

#ifndef TENSORSTORE_INTERNAL_JSON__ARRAY_H_
#define TENSORSTORE_INTERNAL_JSON__ARRAY_H_

/// \file
/// Functions for converting between `::nlohmann::json` and
/// `tensorstore::Array`.

#include <type_traits>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_json {

// Type-erased implmentation of JsonEncodeNestedArray.
::nlohmann::json JsonEncodeNestedArrayImpl(
    ArrayView<const void, dynamic_rank, offset_origin> array,
    absl::FunctionRef<::nlohmann::json(const void*)> encode_element);

// Type-erased implmentation of JsonParseNestedArray.
Result<SharedArray<void>> JsonParseNestedArrayImpl(
    const ::nlohmann::json& j_root, DataType dtype,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& v, void* out)>
        decode_element);

/// Encodes a multi-dimensional array as a nested JSON array.
/// \tparam A An instance of `Array`.
/// \tparam F A function object with a signature compatible with
///     `::nlohmann::json (const A::Element*)`.
/// \param array The array to encode.
/// \param encode_element Function called to encode each element of the array.
/// \returns The nested array JSON representation.
template <typename A, typename F>
std::enable_if_t<IsArray<A>, ::nlohmann::json> JsonEncodeNestedArray(
    const A& array, F encode_element) {
  return JsonEncodeNestedArrayImpl(array, [encode_element](const void* value) {
    return encode_element(static_cast<const typename A::Element*>(value));
  });
}

/// Same as above, but uses `DataType`-defined conversion to json.
Result<::nlohmann::json> JsonEncodeNestedArray(ArrayView<const void> array);

/// Parses a multi-dimensional array from a nested JSON array.
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
      auto array, JsonParseNestedArrayImpl(
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

}  // namespace internal_json
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON__ARRAY_H_
