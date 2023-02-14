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

#include "tensorstore/internal/json/value_as.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

absl::Status ExpectedError(const ::nlohmann::json& j,
                           std::string_view type_name) {
  if (j.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Expected ", type_name, ", but member is missing"));
  }
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Expected ", type_name, ", but received: ", j.dump()));
}

absl::Status ValidationError(const ::nlohmann::json& j,
                             std::string_view type_name) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Validation of ", type_name, " failed, received: ", j.dump()));
}

template <typename T>
absl::Status JsonRequireIntegerImpl<T>::Execute(const ::nlohmann::json& json,
                                                T* result, bool strict,
                                                T min_value, T max_value) {
  if (auto x = JsonValueAs<T>(json, strict)) {
    if (*x >= min_value && *x <= max_value) {
      *result = *x;
      return absl::OkStatus();
    }
  }

  /// NOTE: Eliminate GetTypeName use for integers since it's unreliable
  /// for all the input/output types we want to support.
  constexpr const char* kTypeName = []() {
    if constexpr (sizeof(T) == 4 && std::is_signed_v<T>)
      return "32-bit signed integer";
    if constexpr (sizeof(T) == 4 && std::is_unsigned_v<T>)
      return "32-bit unsigned integer";
    if constexpr (sizeof(T) == 8 && std::is_signed_v<T>)
      return "64-bit signed integer";
    if constexpr (sizeof(T) == 8 && std::is_unsigned_v<T>)
      return "64-bit unsigned integer";
    return GetTypeName(internal::type_identity_t<T>{});
  }();

  if constexpr (kTypeName != nullptr) {
    if (min_value == std::numeric_limits<T>::min() &&
        max_value == std::numeric_limits<T>::max()) {
      return internal_json::ExpectedError(json, kTypeName);
    }
  }
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Expected integer in the range [", min_value, ", ",
                          max_value, "], but received: ", json.dump()));
}
template struct JsonRequireIntegerImpl<std::int64_t>;
template struct JsonRequireIntegerImpl<std::uint64_t>;

template <>
std::optional<std::nullptr_t> JsonValueAs<std::nullptr_t>(
    const ::nlohmann::json& j, bool strict) {
  if (j.is_null()) {
    return nullptr;
  }
  return std::nullopt;
}

template <>
std::optional<bool> JsonValueAs<bool>(const ::nlohmann::json& j, bool strict) {
  if (j.is_boolean()) {
    return j.get<bool>();
  }
  if (!strict && j.is_string()) {
    const auto& str = j.get_ref<std::string const&>();
    if (str == "true") return true;
    if (str == "false") return false;
  }
  return std::nullopt;
}

template <>
std::optional<int64_t> JsonValueAs<int64_t>(const ::nlohmann::json& j,
                                            bool strict) {
  if (j.is_number_unsigned()) {
    auto x = j.get<std::uint64_t>();
    if (x <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
      return static_cast<std::int64_t>(x);
    }
  } else if (j.is_number_integer()) {
    return j.get<std::int64_t>();
  } else if (j.is_number_float()) {
    auto x = j.get<double>();
    if (x >= -9223372036854775808.0 /*=-2^63*/ &&
        x < 9223372036854775808.0 /*=2^63*/ && x == std::floor(x)) {
      return static_cast<std::int64_t>(x);
    }
  } else if (!strict) {
    if (j.is_string()) {
      int64_t result = 0;
      if (absl::SimpleAtoi(j.get_ref<std::string const&>(), &result)) {
        return result;
      }
    }
  }
  return std::nullopt;
}

template <>
std::optional<uint64_t> JsonValueAs<uint64_t>(const ::nlohmann::json& j,
                                              bool strict) {
  if (j.is_number_unsigned()) {
    return j.get<uint64_t>();
  } else if (j.is_number_integer()) {
    int64_t x = j.get<int64_t>();
    if (x >= 0) {
      return static_cast<uint64_t>(x);
    }
  } else if (j.is_number_float()) {
    double x = j.get<double>();
    if (x >= 0.0 && x < 18446744073709551616.0 /*=2^64*/ &&
        x == std::floor(x)) {
      return static_cast<uint64_t>(x);
    }
  } else if (!strict) {
    if (j.is_string()) {
      uint64_t result = 0;
      if (absl::SimpleAtoi(j.get_ref<std::string const&>(), &result)) {
        return result;
      }
    }
  }
  return std::nullopt;
}

template <>
std::optional<double> JsonValueAs<double>(const ::nlohmann::json& j,
                                          bool strict) {
  if (j.is_number()) {
    return j.get<double>();
  }
  if (!strict && j.is_string()) {
    double result = 0;
    if (absl::SimpleAtod(j.get_ref<std::string const&>(), &result)) {
      return result;
    }
  }
  return std::nullopt;
}

template <>
std::optional<std::string> JsonValueAs<std::string>(const ::nlohmann::json& j,
                                                    bool strict) {
  if (j.is_string()) {
    return j.get<std::string>();
  }
  return std::nullopt;
}

}  // namespace internal_json
}  // namespace tensorstore
