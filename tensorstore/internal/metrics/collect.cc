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

#include "tensorstore/internal/metrics/collect.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace tensorstore {
namespace internal_metrics {
namespace {
struct IsNonZero {
  bool operator()(int64_t x) { return x != 0; }
  bool operator()(double x) { return x != 0; }
  bool operator()(const std::string& x) { return !x.empty(); }
  bool operator()(std::monostate) { return false; }
};
}  // namespace

bool IsCollectedMetricNonZero(const CollectedMetric& metric) {
  if (!metric.values.empty()) {
    for (const auto& v : metric.values) {
      if (std::visit(IsNonZero{}, v.value)) return true;
      if (std::visit(IsNonZero{}, v.max_value)) return true;
    }
  } else if (!metric.histograms.empty()) {
    for (const auto& v : metric.histograms) {
      if (v.count != 0) return true;
    }
  }
  return false;
}

void FormatCollectedMetric(
    const CollectedMetric& metric,
    absl::FunctionRef<void(bool has_value, std::string formatted_line)>
        handle_line) {
  std::string field_names;
  if (!metric.field_names.empty()) {
    field_names = absl::StrJoin(metric.field_names, ", ");
  }
  auto metric_name_with_fields = [&](auto& v) -> std::string {
    if (v.fields.empty()) return std::string(metric.metric_name);
    return absl::StrCat(metric.metric_name, "<", field_names, ">[",
                        absl::StrJoin(v.fields, ", "), "]");
  };

  if (!metric.values.empty()) {
    for (auto& v : metric.values) {
      bool has_value = false;
      std::string line = metric_name_with_fields(v);
      if (std::holds_alternative<std::monostate>(v.max_value)) {
        // A Normal value.
        std::visit(
            [&](auto x) {
              has_value |= IsNonZero{}(x);
              absl::StrAppend(&line, "=", x);
            },
            v.value);
      } else {
        // A Gauge-like value.
        std::visit(
            [&](auto x) {
              has_value |= IsNonZero{}(x);
              absl::StrAppend(&line, "={value=", x);
            },
            v.value);
        std::visit(
            [&](auto x) {
              has_value |= IsNonZero{}(x);
              if constexpr (!std::is_same<decltype(x), std::monostate>::value) {
                absl::StrAppend(&line, ", max=", x, "}");
              }
            },
            v.max_value);
      }
      handle_line(has_value, std::move(line));
    }
  }
  if (!metric.histograms.empty()) {
    for (auto& v : metric.histograms) {
      std::string line = metric_name_with_fields(v);
      absl::StrAppend(&line, "={count=", v.count, " mean=", v.mean,
                      " buckets=[");

      // find the last bucket with data.
      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) end--;

      // element 0 is typically the underflow bucket.
      auto it = v.buckets.begin();
      if (end > 0) {
        absl::StrAppend(&line, *it);
      }
      // every 10 elements insert an extra space.
      for (size_t i = 1; i < end;) {
        size_t j = std::min(i + 10, end);
        absl::StrAppend(&line, ",  ");
        absl::StrAppend(&line, absl::StrJoin(it + i, it + j, ","));
        i = j;
      }
      absl::StrAppend(&line, "]}");
      handle_line(/*has_value=*/v.count, std::move(line));
    }
  }
}

/// Converts a CollectedMetric to json.
::nlohmann::json CollectedMetricToJson(const CollectedMetric& metric) {
  ::nlohmann::json::object_t result;
  result["name"] = metric.metric_name;

  auto set_field_keys = [&](auto& v, ::nlohmann::json::object_t& h) {
    assert(metric.field_names.size() == v.fields.size());
    for (size_t i = 0; i < metric.field_names.size(); ++i) {
      if (metric.field_names[i] == "value" ||
          metric.field_names[i] == "count" ||
          metric.field_names[i] == "max_value" ||
          metric.field_names[i] == "sum") {
        h[absl::StrCat("_", metric.field_names[i])] = v.fields[i];
      } else {
        h[std::string(metric.field_names[i])] = v.fields[i];
      }
    }
  };

  std::vector<::nlohmann::json> values;
  if (!metric.values.empty()) {
    for (const auto& v : metric.values) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      std::visit([&tmp](auto x) { tmp["value"] = x; }, v.value);
      if (!std::holds_alternative<std::monostate>(v.max_value)) {
        std::visit(
            [&tmp](auto x) {
              if constexpr (!std::is_same<decltype(x), std::monostate>::value) {
                tmp["max_value"] = x;
              }
            },
            v.max_value);
      }
      values.push_back(std::move(tmp));
    }
  } else if (!metric.histograms.empty()) {
    for (const auto& v : metric.histograms) {
      ::nlohmann::json::object_t tmp{};
      set_field_keys(v, tmp);
      tmp["count"] = v.count;
      tmp["mean"] = v.mean;
      tmp["sum_of_squared_deviation"] = v.sum_of_squared_deviation;

      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) end--;

      auto it = v.buckets.begin();
      for (size_t i = 0; i < end; ++i) {
        tmp[absl::StrCat(i)] = *it++;
      }
      values.push_back(std::move(tmp));
    }
  }
  result["values"] = std::move(values);
  return result;
}

}  // namespace internal_metrics
}  // namespace tensorstore
