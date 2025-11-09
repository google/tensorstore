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

#ifndef TENSORSTORE_INTERNAL_JSON_METADATA_MATCHING_H_
#define TENSORSTORE_INTERNAL_JSON_METADATA_MATCHING_H_

/// \file
///
/// Functions for producing JSON metadata mismatch error messages, and for
/// validating that a JSON object is a subset of another JSON object.

#include <string_view>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

/// Returns a failed precondition status indicating a member name and JSON
/// values that did not match.
template <typename T, typename U>
absl::Status MetadataMismatchError(std::string_view name, const T& expected,
                                   const U& actual) {
  return absl::FailedPreconditionError(
      tensorstore::StrCat("Expected ", tensorstore::QuoteString(name), " of ",
                          ::nlohmann::json(expected).dump(),
                          " but received: ", ::nlohmann::json(actual).dump()));
}

/// Validates that the (key, value) pairs of `expected` are a subset of the
/// (key, value) pairs of `actual`.
absl::Status ValidateMetadataSubset(const ::nlohmann::json::object_t& expected,
                                    const ::nlohmann::json::object_t& actual);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_METADATA_MATCHING_H_
