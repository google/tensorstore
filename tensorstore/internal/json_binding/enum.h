// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_ENUM_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_ENUM_H_

#include <stddef.h>

#include <string>
#include <utility>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

/// Returns a `Binder` for enum-like types.
///
/// Example usage:
///
///     enum class TestEnum { a, b };
///     const auto binder = jb::Enum<TestEnum, std::string_view>({
///         {TestEnum::a, "a"},
///         {TestEnum::b, "b"},
///     });
///
/// When converting to JSON, equality comparison is used for the `EnumValue`
/// values.  When converting from JSON, `JsonSame` is used for comparison.
///
/// \tparam EnumValue The C++ enum-like type to bind.  May be any regular type
///     that supports equality comparison.
/// \tparam JsonValue The JSON value representation, may be
/// `std::string_view`,
///     `int`, or another type convertible to `::nlohmann::json`.
/// \param values Array of `EnumValue`/`JsonValue` pairs.
template <typename EnumValue, typename JsonValue, std::size_t N>
constexpr auto Enum(const std::pair<EnumValue, JsonValue> (&values)[N]) {
  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    for (const auto& p : values) {
      if constexpr (is_loading) {
        if (internal_json::JsonSame(p.second, *j)) {
          *obj = p.first;
          return absl::OkStatus();
        }
      } else {
        if (p.first == *obj) {
          *j = p.second;
          return absl::OkStatus();
        }
      }
    }
    if constexpr (is_loading) {
      return internal_json::ExpectedError(
          *j,
          tensorstore::StrCat(
              "one of ",
              absl::StrJoin(values, ", ", [](std::string* out, const auto& p) {
                *out += ::nlohmann::json(p.second).dump();
              })));
    } else {
      ABSL_UNREACHABLE();  // COV_NF_LINE
    }
  };
}

/// Provides a `Binder` which maps pairs of {c++ object, json value}.
/// This may be useful for std::variant types.
///
/// The provided values are copied, and not converted to json when called, so
/// reference-like types should be avoided.
///
/// Example usage:
///
///     const auto binder = jb::MapValue(DefaultBinder<>,
///                                      std::make_pair(Dog{}, "dog"),
///                                      std::make_pair(Cat{}, "cat));
///
/// When converting to JSON, equality comparison is used for the `Value`
/// values.  When converting from JSON, `JsonSame` is used for comparison.
///
/// \tparam Binder  Default binder type.
/// \param binder   The default binder value. Required.
///     May be `jb::DefaultBinder<>`
///
/// \tparam Value  A C++ value representation.  May be any regular type that
///     supports equality comparison and assignment.
/// \tparam JsonValue  A JSON value representation, may be `char *`, `int`, or
///     another type convertible to `::nlohmann::json`.
/// \param pairs   Array of std::pair<Value, JsonValue> for each mapping.
template <typename Binder, typename... Value, typename... JsonValue>
constexpr auto MapValue(Binder binder, std::pair<Value, JsonValue>... pairs) {
  constexpr size_t N = sizeof...(pairs);
  static_assert(N > 0);

  return [=](auto is_loading, const auto& options, auto* obj,
             auto* j) -> absl::Status {
    if constexpr (is_loading) {
      if (((internal_json::JsonSame(*j, pairs.second) &&
            (static_cast<void>(*obj = pairs.first), true)) ||
           ...))
        return absl::OkStatus();
    } else {
      if ((((*obj == pairs.first) &&
            (static_cast<void>(*j = pairs.second), true)) ||
           ...))
        return absl::OkStatus();
    }
    return binder(is_loading, options, obj, j);
  };
}

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_ENUM_H_
