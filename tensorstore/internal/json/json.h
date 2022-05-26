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

#ifndef TENSORSTORE_INTERNAL_JSON__JSON_H_
#define TENSORSTORE_INTERNAL_JSON__JSON_H_

#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_json {

// ParseJson wraps the ::nlohmann::json::parse calls to avoid throwing
// exceptions.
::nlohmann::json ParseJson(std::string_view str);

/// Parses a JSON array.
///
/// \param j The JSON value to parse.
/// \param size_callback Callback invoked with the array size before parsing any
///     elements.  Parsing stops if it returns an error.
/// \param element_callback Callback invoked for each array element after
///     `size_callback` has been invoked.  Parsing stops if it returns an error.
/// \returns `absl::Status()` on success, or otherwise the first error returned
/// by
///     `size_callback` or `element_callback`.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not an array.
absl::Status JsonParseArray(
    const ::nlohmann::json& j,
    absl::FunctionRef<absl::Status(std::ptrdiff_t size)> size_callback,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& value,
                                   std::ptrdiff_t index)>
        element_callback);

/// Validates that `parsed_size` matches `expected_size`.
///
/// If the sizes don't match, returns a `absl::Status` with an informative error
/// message.
///
/// This function is particularly useful to call from a `size_callback` passed
/// to `JsonParseArray`.
///
/// \param parsed_size Parsed size of array.
/// \param expected_size Expected size of array.
/// \returns `absl::Status()` if `parsed_size == expected_size`.
/// \error `absl::StatusCode::kInvalidArgument` if `parsed_size !=
///     expected_size`.
absl::Status JsonValidateArrayLength(std::ptrdiff_t parsed_size,
                                     std::ptrdiff_t expected_size);

/// Removes the specified member from `*j_obj` if it is present.
///
/// \returns The extracted member if present, or
///     `::nlohmann::json::value_t::discarded` if not present.
::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   std::string_view name);

/// Returns an error indicating that all members of `j_obj` are unexpected.
absl::Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj);

/// Returns `true` if `a` and `b` are equal.
///
/// Unlike `operator==`, the comparison is non-recursive, and is therefore safe
/// from stack overflow even for deeply nested structures.
///
/// Like `operator==`, two int64_t/uint64_t/double values representing exactly
/// the same number are all considered equal even if their types differ.
///
/// Unlike `operator==`, two `discarded` values are considered equal.
bool JsonSame(const ::nlohmann::json& a, const ::nlohmann::json& b);

}  // namespace internal_json
namespace internal {

using internal_json::ParseJson;

}
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON__JSON_H_
