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

#ifndef TENSORSTORE_INTERNAL_JSON__VALUE_AS_H_
#define TENSORSTORE_INTERNAL_JSON__VALUE_AS_H_

/// \file
///
/// Low-level functions for extracting primitive values from JSON values.
///
/// In most cases the higher-level json_binding interfaces, implemented in terms
/// of these functions, should be used instead.

// This is extracted from json.h to avoid circular/excessive dependencies.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_json {

/// Retuns an error message for a json value with the expected type.
absl::Status ExpectedError(const ::nlohmann::json& j,
                           std::string_view type_name);

/// Returns an error message indicating that json field validation failed.
absl::Status ValidationError(const ::nlohmann::json& j,
                             std::string_view type_name);

/// GetTypeName returns the expected json field type name for error messages.
inline constexpr const char* GetTypeName(
    internal::type_identity<std::int64_t>) {
  return "64-bit signed integer";
}
inline constexpr const char* GetTypeName(
    internal::type_identity<std::uint64_t>) {
  return "64-bit unsigned integer";
}
inline constexpr const char* GetTypeName(
    internal::type_identity<std::int32_t>) {
  return "32-bit signed integer";
}
inline constexpr const char* GetTypeName(
    internal::type_identity<std::uint32_t>) {
  return "32-bit unsigned integer";
}
inline constexpr const char* GetTypeName(internal::type_identity<double>) {
  return "64-bit floating-point number";
}
inline constexpr const char* GetTypeName(internal::type_identity<std::string>) {
  return "string";
}
inline constexpr const char* GetTypeName(internal::type_identity<bool>) {
  return "boolean";
}
inline constexpr const char* GetTypeName(
    internal::type_identity<std::nullptr_t>) {
  return "null";
}
inline constexpr const char* GetTypeName(...) { return nullptr; }

/// Implementation of `JsonRequireInteger` defined below.
///
/// Only defined for `T` equal to `std::int64_t` or `std::uint64_t`.
///
/// Defined as a class to simplify explicit instantiation.
template <typename T>
struct JsonRequireIntegerImpl {
  static absl::Status Execute(const ::nlohmann::json& json, T* result,
                              bool strict, T min_value, T max_value);
};

/// Return a json object as the C++ type, if conversion is possible.
/// Will attempt to parse strings and perform exact-numeric conversions
/// between the json data value and the c++ type.
///
/// For <bool>, accepted string formats are:  "true", "false"
/// For <double>, accepted string formats are those supported by std::strtod.
/// For <int64_t, uint64_t>, accepted string formats are base-10 numbers,
///   e.g. (-+)?[0-9]+, with the restriction that the value must fit into a
///   64-bit integer and uint64_t cannot include a negative prefix.
///
/// \param json The value to parse.
/// \param strict If `true`, string conversions are not permitted.
template <typename T>
std::optional<T> JsonValueAs(const ::nlohmann::json& j, bool strict = false) {
  static_assert(!std::is_same_v<T, T>, "Target type not supported.");
}

template <>
std::optional<std::nullptr_t> JsonValueAs<std::nullptr_t>(
    const ::nlohmann::json& j, bool strict);

template <>
std::optional<bool> JsonValueAs<bool>(const ::nlohmann::json& j, bool strict);

template <>
std::optional<int64_t> JsonValueAs<int64_t>(const ::nlohmann::json& j,
                                            bool strict);

template <>
std::optional<uint64_t> JsonValueAs<uint64_t>(const ::nlohmann::json& j,
                                              bool strict);

template <>
std::optional<double> JsonValueAs<double>(const ::nlohmann::json& j,
                                          bool strict);

template <>
std::optional<std::string> JsonValueAs<std::string>(const ::nlohmann::json& j,
                                                    bool strict);

/// JsonRequireValueAs attempts to convert and assign the json object to the
/// result value.
///
/// \param j The JSON value to parse
/// \param result A pointer to an assigned value (may be nullptr)
/// \param is_valid A function or lambda taking T, returning true when
///     valid.
/// \param strict If `true`, string conversions are not permitted.
template <typename T, typename ValidateFn>
std::enable_if_t<!std::is_same_v<ValidateFn, bool>, absl::Status>
JsonRequireValueAs(const ::nlohmann::json& j, T* result, ValidateFn is_valid,
                   bool strict = false) {
  auto value = JsonValueAs<T>(j, strict);
  if (!value) {
    return internal_json::ExpectedError(
        j, internal_json::GetTypeName(internal::type_identity<T>{}));
  }
  // NOTE: improve is_valid; allow bool or status returns.
  if (!is_valid(*value)) {
    return internal_json::ValidationError(
        j, internal_json::GetTypeName(internal::type_identity<T>{}));
  }
  if (result != nullptr) {
    *result = std::move(*value);
  }
  return absl::OkStatus();
}

/// As above, omitting the validation function.
template <typename T>
absl::Status JsonRequireValueAs(const ::nlohmann::json& j, T* result,
                                bool strict = false) {
  return JsonRequireValueAs(
      j, result, [](const T&) { return true; }, strict);
}

/// Attempts to convert `json` to an integer in the range
/// `[min_value, max_value]`.
///
/// \tparam T A built-in signed or unsigned integer type.
/// \param json The JSON value.
/// \param result[out] Non-null pointer to location where result is stored on
///     success.
/// \param strict If `true`, conversions from string are not permitted.
/// \param min_value The minimum allowed value (inclusive).
/// \param max_value The maximum allowed value (inclusive).
/// \returns `absl::Status()` if successful.
/// \error `absl::StatusCode::kInvalidArgument` on failure.
template <typename T>
absl::Status JsonRequireInteger(
    const ::nlohmann::json& json, T* result, bool strict = false,
    internal::type_identity_t<T> min_value = std::numeric_limits<T>::min(),
    internal::type_identity_t<T> max_value = std::numeric_limits<T>::max()) {
  static_assert(std::is_signed_v<T> || std::is_unsigned_v<T>,
                "T must be an integer type.");
  using U =
      std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
  U temp;
  auto status = internal_json::JsonRequireIntegerImpl<U>::Execute(
      json, &temp, strict, min_value, max_value);
  if (status.ok()) *result = temp;
  return status;
}

}  // namespace internal_json
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON__VALUE_AS_H_
