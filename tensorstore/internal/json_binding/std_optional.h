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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STD_OPTIONAL_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STD_OPTIONAL_H_

#include <optional>

#include "absl/status/status.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"

namespace tensorstore {
namespace internal_json_binding {

/// Returns a `Binder` for `std::optional<T>`.
///
/// When loading, if the JSON value is equal to `nullopt_to_json()`, sets the
/// optional object to `std::nullopt`.  Otherwise, constructs the object and
/// calls `value_binder`.
///
/// When saving, if the optional object is equal to `std::nullopt`, sets the
/// JSON value to `nullopt_to_json()`.  Otherwise, calls `value_binder`.
///
/// \param value_binder `Binder` for `T` to use in the case the value is
///     present.
/// \param nullopt_to_json Function with signature `::nlohmann::json ()` that
///     returns the JSON representation for `std::nullopt`.
template <typename ValueBinder, typename NulloptToJson>
constexpr auto Optional(ValueBinder value_binder,
                        NulloptToJson nullopt_to_json) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) -> absl::Status {
    if constexpr (is_loading) {
      ::nlohmann::json nullopt_json = nullopt_to_json();
      if (internal_json::JsonSame(*j, nullopt_json)) {
        // std::optional is default-initialized as std::nullopt;
        // Assume other types are likewise default-initialized.
        return absl::OkStatus();
      } else {
        return value_binder(is_loading, options, &obj->emplace(), j);
      }
    } else {
      if (obj->has_value()) {
        return value_binder(is_loading, options, &**obj, j);
      } else {
        *j = nullopt_to_json();
        return absl::OkStatus();
      }
    }
  };
}

/// Same as above, but converts `std::optional` to
/// ::nlohmann::json::value_t::discarded` (for use with `Member`).
template <typename ValueBinder = decltype(DefaultBinder<>)>
constexpr auto Optional(ValueBinder value_binder = DefaultBinder<>) {
  return Optional(value_binder, [] {
    return ::nlohmann::json(::nlohmann::json::value_t::discarded);
  });
}

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace optional_binder {
constexpr inline auto OptionalBinder = [](auto is_loading, const auto& options,
                                          auto* obj, ::nlohmann::json* j) {
  return Optional()(is_loading, options, obj, j);
};
}  // namespace optional_binder
using optional_binder::OptionalBinder;

/// Registers `Optional` as the default binder for `std::optional`.
template <typename T>
inline constexpr auto& DefaultBinder<std::optional<T>> = OptionalBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_STD_OPTIONAL_H_
