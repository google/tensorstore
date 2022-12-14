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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_ABSL_TIME_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_ABSL_TIME_H_

#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_fwd.h"

namespace tensorstore {
namespace internal_json_binding {

namespace rfc3339_time_binder {
constexpr inline auto Rfc3339TimeBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    if (!j->is_string()) {
      return internal_json::ExpectedError(*j,
                                          "Date formatted as RFC3339 string");
    }
    std::string error;
    if (absl::ParseTime(absl::RFC3339_full, j->get_ref<std::string const&>(),
                        obj, &error)) {
      return absl::OkStatus();
    }
    return internal_json::ExpectedError(*j, "Date formatted as RFC3339 string");
  } else {
    *j = absl::FormatTime(*obj, absl::UTCTimeZone());
    return absl::OkStatus();
  }
};
}  // namespace rfc3339_time_binder
using rfc3339_time_binder::Rfc3339TimeBinder;

template <>
constexpr inline auto DefaultBinder<absl::Time> = Rfc3339TimeBinder;

namespace duration_binder {
constexpr inline auto DurationBinder = [](auto is_loading, const auto& options,
                                          auto* obj,
                                          ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    if (j->is_string()) {
      std::string error;
      if (absl::ParseDuration(j->get_ref<std::string const&>(), obj)) {
        return absl::OkStatus();
      }
    }
    return internal_json::ExpectedError(
        *j,
        R"(Duration formatted as a string using time units "ns", "us" "ms", "s", "m", or "h".)");
  } else {
    *j = absl::FormatDuration(*obj);
    return absl::OkStatus();
  }
};

}  // namespace duration_binder
using duration_binder::DurationBinder;

template <>
constexpr inline auto DefaultBinder<absl::Duration> = DurationBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_ABSL_TIME_H_
