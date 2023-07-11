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

#ifndef TENSORSTORE_UTIL_JSON_ABSL_FLAG_H_
#define TENSORSTORE_UTIL_JSON_ABSL_FLAG_H_

#include <type_traits>

#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/bindable.h"

namespace tensorstore {

/// Wraps a JSON-bindable type for use as an `ABSL_FLAG` type.
///
/// Declaration example:
///   ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, spec,
///             {}, "tensorstore JSON specification");
///
/// Use example:
///   auto ts = tensorstore::Open(absl::GetFlag(FLAGS_spec).value);
///
/// \tparam T Type that supports JSON serialization.
template <typename T>
struct JsonAbslFlag {
  T value;

  JsonAbslFlag() = default;

  template <typename... U,
            typename = std::enable_if_t<std::is_constructible_v<T, U&&...>>>
  JsonAbslFlag(U&&... arg) : value(std::forward<U>(arg)...) {}

  friend std::string AbslUnparseFlag(const JsonAbslFlag& json_flag) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto j, internal_json_binding::ToJson(json_flag.value), "");
    if (j.is_discarded()) return {};
    return absl::UnparseFlag(j.dump());
  }

  friend bool AbslParseFlag(std::string_view in, JsonAbslFlag* out,
                            std::string* error) {
    if (in.empty()) {
      // empty string yields a default-constructed flag value.
      out->value = {};
      return true;
    }

    ::nlohmann::json j = ::nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded()) {
      *error = absl::StrFormat("Failed to parse JSON: '%s'", in);
      return false;
    }
    absl::Status status = internal_json_binding::DefaultBinder<>(
        std::true_type{}, internal_json_binding::NoOptions{}, &out->value, &j);
    if (!status.ok()) {
      *error = absl::StrFormat("Failed to bind JSON: %s", status.message());
      return false;
    }
    return true;
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_JSON_ABSL_FLAG_H_
