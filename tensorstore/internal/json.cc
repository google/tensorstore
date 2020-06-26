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

#include "tensorstore/internal/json.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/util/function_view.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

bool JsonSame(const ::nlohmann::json& a, const ::nlohmann::json& b) {
  if (a == b) return true;
  if (a.is_discarded() && b.is_discarded()) return true;
  return false;
}

Status ExpectedError(const ::nlohmann::json& j, absl::string_view type_name) {
  if (j.is_discarded()) {
    return absl::InvalidArgumentError(
        StrCat("Expected ", type_name, ", but member is missing"));
  }
  return absl::InvalidArgumentError(
      StrCat("Expected ", type_name, ", but received: ", j.dump()));
}

Status ValidationError(const ::nlohmann::json& j, absl::string_view type_name) {
  return absl::InvalidArgumentError(
      StrCat("Validation of ", type_name, " failed, received: ", j.dump()));
}

Status MaybeAnnotateMemberError(const Status& status,
                                absl::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(
      status, StrCat("Error parsing object member ", QuoteString(member_name)));
}

Status MaybeAnnotateMemberConvertError(const Status& status,
                                       absl::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(status, StrCat("Error converting object member ",
                                            QuoteString(member_name)));
}

Status MaybeAnnotateArrayElementError(const Status& status, std::size_t i,
                                      bool is_loading) {
  return MaybeAnnotateStatus(
      status,
      tensorstore::StrCat("Error ", is_loading ? "parsing" : "converting",
                          " value at position ", i));
}

}  // namespace internal_json
namespace internal {

::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   absl::string_view name) {
  if (auto it = j_obj->find(name); it != j_obj->end()) {
    auto node = j_obj->extract(it);
    return std::move(node.mapped());
  }
  return ::nlohmann::json(::nlohmann::json::value_t::discarded);
}
Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj) {
  return absl::InvalidArgumentError(
      StrCat("Object includes extra members: ",
             absl::StrJoin(j_obj, ",", [](std::string* out, const auto& p) {
               *out += QuoteString(p.first);
             })));
}

// ParseJson wraps the ::nlohmann::json::parse calls to avoid throwing
// exceptions.
::nlohmann::json ParseJson(absl::string_view str) {
  return ::nlohmann::json::parse({str.begin(), str.end()}, nullptr, false);
}

template <>
absl::optional<bool> JsonValueAs<bool>(const ::nlohmann::json& j, bool strict) {
  if (j.is_boolean()) {
    return j.get<bool>();
  }
  if (!strict && j.is_string()) {
    const auto& str = j.get_ref<std::string const&>();
    if (str == "true") return true;
    if (str == "false") return false;
  }
  return absl::nullopt;
}

template <>
absl::optional<int64_t> JsonValueAs<int64_t>(const ::nlohmann::json& j,
                                             bool strict) {
  if (j.is_number_unsigned()) {
    auto x = j.get<std::uint64_t>();
    if (x <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
      return static_cast<std::int64_t>(x);
    }
  } else if (j.is_number_integer()) {
    return j.get<std::int64_t>();
  } else if (!strict) {
    if (j.is_number_float()) {
      auto x = j.get<double>();
      if (x >= -9223372036854775808.0 /*=-2^63*/ &&
          x < 9223372036854775808.0 /*=2^63*/ && x == std::floor(x)) {
        return static_cast<std::int64_t>(x);
      }
    } else if (j.is_string()) {
      int64_t result = 0;
      if (absl::SimpleAtoi(j.get_ref<std::string const&>(), &result)) {
        return result;
      }
    }
  }
  return absl::nullopt;
}

template <>
absl::optional<uint64_t> JsonValueAs<uint64_t>(const ::nlohmann::json& j,
                                               bool strict) {
  if (j.is_number_unsigned()) {
    return j.get<uint64_t>();
  } else if (j.is_number_integer()) {
    int64_t x = j.get<int64_t>();
    if (x >= 0) {
      return static_cast<uint64_t>(x);
    }
  } else if (!strict) {
    // Fallback numeric parsing.
    if (j.is_number_float()) {
      double x = j.get<double>();
      if (x >= 0.0 && x < 18446744073709551616.0 /*=2^64*/ &&
          x == std::floor(x)) {
        return static_cast<uint64_t>(x);
      }
    }
    if (j.is_string()) {
      uint64_t result = 0;
      if (absl::SimpleAtoi(j.get_ref<std::string const&>(), &result)) {
        return result;
      }
    }
  }
  return absl::nullopt;
}

template <>
absl::optional<double> JsonValueAs<double>(const ::nlohmann::json& j,
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
  return absl::nullopt;
}

template <>
absl::optional<std::string> JsonValueAs<std::string>(const ::nlohmann::json& j,
                                                     bool strict) {
  if (j.is_string()) {
    return j.get<std::string>();
  }
  return absl::nullopt;
}

Status JsonParseArray(
    const ::nlohmann::json& j,
    FunctionView<Status(std::ptrdiff_t size)> size_callback,
    FunctionView<Status(const ::nlohmann::json& value, std::ptrdiff_t index)>
        element_callback) {
  const auto* j_array = j.get_ptr<const ::nlohmann::json::array_t*>();
  if (!j_array) {
    return internal_json::ExpectedError(j, "array");
  }
  const std::ptrdiff_t size = j_array->size();
  TENSORSTORE_RETURN_IF_ERROR(size_callback(size));
  for (DimensionIndex i = 0; i < size; ++i) {
    auto status = element_callback(j[i], i);
    if (!status.ok()) {
      return MaybeAnnotateStatus(status,
                                 StrCat("Error parsing value at position ", i));
    }
  }
  return absl::OkStatus();
}

Status JsonValidateArrayLength(std::ptrdiff_t parsed_size,
                               std::ptrdiff_t expected_size) {
  if (parsed_size != expected_size) {
    return absl::InvalidArgumentError(StrCat("Array has length ", parsed_size,
                                             " but should have length ",
                                             expected_size));
  }
  return absl::OkStatus();
}

Status JsonValidateObjectMembers(
    const ::nlohmann::json::object_t& j,
    span<const absl::string_view> allowed_members) {
  std::vector<std::string> extra_members;
  const auto find_member =
      [&](const ::nlohmann::json::object_t::value_type& p) {
        for (absl::string_view member : allowed_members) {
          if (member == p.first) return;
        }
        extra_members.push_back(QuoteString(p.first));
      };
  for (const auto& p : j) {
    find_member(p);
  }
  if (!extra_members.empty()) {
    return absl::InvalidArgumentError(StrCat(
        "Object includes extra members: ", absl::StrJoin(extra_members, ",")));
  }
  return absl::OkStatus();
}

Status JsonValidateObjectMembers(
    const ::nlohmann::json& j, span<const absl::string_view> allowed_members) {
  if (const auto* obj = j.get_ptr<const ::nlohmann::json::object_t*>()) {
    return JsonValidateObjectMembers(*obj, allowed_members);
  }
  return internal_json::ExpectedError(j, "object");
}

Status JsonRequireObjectMember(
    const ::nlohmann::json::object_t& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handle) {
  auto it = j.find(member_name);
  if (it == j.end()) {
    return absl::InvalidArgumentError(
        StrCat("Missing object member ", QuoteString(member_name)));
  }
  return internal_json::MaybeAnnotateMemberError(handle(it->second),
                                                 member_name);
}

Status JsonRequireObjectMember(
    const ::nlohmann::json& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handle) {
  if (const auto* obj = j.get_ptr<const ::nlohmann::json::object_t*>()) {
    return JsonRequireObjectMember(*obj, member_name, handle);
  }
  return internal_json::ExpectedError(j, "object");
}

Status JsonHandleObjectMember(
    const ::nlohmann::json::object_t& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handle) {
  auto it = j.find(member_name);
  if (it == j.end()) {
    return absl::OkStatus();
  }
  return internal_json::MaybeAnnotateMemberError(handle(it->second),
                                                 member_name);
}

Status JsonHandleObjectMember(
    const ::nlohmann::json& j, const char* member_name,
    FunctionView<Status(const ::nlohmann::json&)> handle) {
  if (const auto* obj = j.get_ptr<const ::nlohmann::json::object_t*>()) {
    return JsonHandleObjectMember(*obj, member_name, handle);
  }
  return internal_json::ExpectedError(j, "object");
}

}  // namespace internal

namespace internal_json {
template <typename T>
Status JsonRequireIntegerImpl<T>::Execute(const ::nlohmann::json& json,
                                          T* result, bool strict, T min_value,
                                          T max_value) {
  if (auto x = internal::JsonValueAs<T>(json, strict)) {
    if (*x >= min_value && *x <= max_value) {
      *result = *x;
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(StrCat("Expected integer in the range [",
                                           min_value, ", ", max_value,
                                           "], but received: ", json.dump()));
}
template struct JsonRequireIntegerImpl<std::int64_t>;
template struct JsonRequireIntegerImpl<std::uint64_t>;

}  // namespace internal_json
}  // namespace tensorstore
