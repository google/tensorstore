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

#ifndef TENSORSTORE_INTERNAL_JSON_OBJECT_WITH_TYPE_H_
#define TENSORSTORE_INTERNAL_JSON_OBJECT_WITH_TYPE_H_

#include <iosfwd>
#include <string>

#include <nlohmann/json.hpp>
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Represents a JSON object with a string `"$type"` member.
///
/// It is intended that the `type` member specifies a registered driver/handler,
/// to which the `object` is passed as a set of options.
struct JsonObjectWithType {
  JsonObjectWithType() = default;

  /// Constructs from the specified `type` and `object`.
  ///
  /// \dchecks `object.count("$type") == 0`.
  JsonObjectWithType(std::string type, ::nlohmann::json::object_t object = {});

  /// Value of the `"$type"` member.
  std::string type;

  /// Object members excluding `"$type"`.
  ::nlohmann::json::object_t object;

  /// Parses a JSON value.
  ///
  /// \returns The parsed representation on success.
  /// \error `absl::StatusCode::kInvalidArgument` if `json` is not a JSON object
  ///     or does not contain a string `"$type"` member.
  static Result<JsonObjectWithType> Parse(::nlohmann::json json);

  /// Converts `x` back to a JSON object with `"$type"` member.
  friend void to_json(::nlohmann::json& out,  // NOLINT
                      JsonObjectWithType x);

  /// Compares the `type` and `object` for equality.
  friend bool operator==(const JsonObjectWithType& a,
                         const JsonObjectWithType& b);
  friend bool operator!=(const JsonObjectWithType& a,
                         const JsonObjectWithType& b) {
    return !(a == b);
  }

  /// Equivalent to `os << ::nlohmann::json(x).dump()`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const JsonObjectWithType& x);
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_OBJECT_WITH_TYPE_H_
