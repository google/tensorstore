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

#ifndef TENSORSTORE_INTERNAL_JSON_POINTER_H_
#define TENSORSTORE_INTERNAL_JSON_POINTER_H_

/// \file
/// JSON Pointer operations.
///
/// JSON Pointer is defined by RFC 6901 (https://tools.ietf.org/html/rfc6901).
///
/// Note that ::nlohmann::json includes support for JSON Pointer, but that
/// support is not used here because ::nlohmann::json provides no way to handle
/// JSON Pointer-related errors when exceptions are disabled, and it lacks
/// necessary functionality for implementing the `JsonChangeMap` class required
/// by the "json" TensorStore driver.

#include <string_view>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace json_pointer {

/// Validates that `s` is a JSON Pointer that conforms to RFC 6901
/// (https://tools.ietf.org/html/rfc6901#section-3)
///
/// This does not validate that `s` is valid UTF-8, but even if passed invalid
/// UTF-8, still validates all other constraints of the JSON Pointer spec.
absl::Status Validate(std::string_view s);

enum CompareResult {
  // `a` is less than `b`
  kLessThan = -2,
  // `a` contains `b`
  kContains = -1,
  // `a` equals `b`
  kEqual = 0,
  // `a` is contained in `b`
  kContainedIn = 1,
  // `a` is greater than `b`
  kGreaterThan = 2,
};

/// Compares JSON Pointers
///
/// Comparison is lexicographical over the sequence of split, decoded reference
/// tokens.  Note that all reference tokens are compared as strings, even
/// reference tokens that are syntactically valid base-10 numbers (and may be
/// intended to be used to index an array).
///
/// \pre `Validate(a).ok()`
/// \pre `Validate(b).ok()`
CompareResult Compare(std::string_view a, std::string_view b);

/// Encodes a string as a JSON Pointer reference token.
///
/// Replaces '~' with "~0" and '/' with "~1".
std::string EncodeReferenceToken(std::string_view token);

enum DereferenceMode {
  /// If the pointer does not refer to an existing value, returns an error.
  kMustExist,
  /// The existing value is modified in place to add any necessary missing
  /// object members, and to append an array element if "-" is specified as the
  /// reference token.  Any `discarded` values within the access path are
  /// converted to objects.
  kCreate,
  /// Performs the same checks as for `kCreate`, but never modifies the existing
  /// value.  Instead, returns `nullptr` if the existing value would have been
  /// modified.
  kSimulateCreate,
  /// Behaves like `kSimulateCreate`, except that if the leaf value is present,
  /// it is deleted from the existing value.  The return value is always
  /// `nullptr`.
  kDelete,
};

/// Returns a pointer to the value referenced by `sub_value_pointer`.
/// \pre `Validate(sub_value_pointer).ok()`
/// \param full_value The JSON value to access.
/// \param sub_value_pointer JSON Pointer into `full_value`.
/// \param mode Specifies how missing members are handled.
/// \returns If `mode == kDelete`, `nullptr`.  Otherwise, on success, non-null
///     pointer to value within `full_value` referred to by `sub_value_pointer`.
///     If `mode == kSimulateCreate` and `sub_value_pointer` refers to a missing
///     object/array member that can be created, returns `nullptr`.
/// \error `absl::StatusCode::kNotFound` if `mode == kMustExist` and
///     `sub_value_pointer` specifies a missing member (or "-" for an array).
/// \error `absl::StatusCode::kFailedPrecondition` if an ancestor value resolves
///     to a non-array, non-object value (if `mode == kMustExist`) or a
///     non-array, non-object, non-discarded value (if `mode == kCreate` or
///     `mode == kSimulateCreate`).
/// \error `absl::StatusCode::kFailedPrecondition` if `sub_value_pointer`
///     specifies an invalid reference token for an array value.
Result<::nlohmann::json*> Dereference(::nlohmann::json& full_value,
                                      std::string_view sub_value_pointer,
                                      DereferenceMode mode);

/// Same as above, but for const values.
///
/// \param full_value The JSON value to access.
/// \param sub_value_pointer JSON Pointer into `full_value`.
/// \param mode Specifies the dereference mode, must be `kMustExist` or
///     `kSimulateCreate`.
/// \pre `Validate(sub_value_pointer).ok()`
Result<const ::nlohmann::json*> Dereference(const ::nlohmann::json& full_value,
                                            std::string_view sub_value_pointer,
                                            DereferenceMode mode = kMustExist);

/// Replaces the value in `full_value` specified by `sub_value_pointer` with
/// `new_sub_value`.
///
/// \param full_value The JSON value to modify.
/// \param sub_value_pointer JSON Pointer into `full_value`.
/// \param new_sub_value New value to set in `full_value`.  If equal to
///     `::nlohmann::json::value_t::discarded`, then the value indicated by
///     `sub_value_pointer` is deleted.
absl::Status Replace(::nlohmann::json& full_value,
                     std::string_view sub_value_pointer,
                     ::nlohmann::json new_sub_value);

}  // namespace json_pointer
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_POINTER_H_
