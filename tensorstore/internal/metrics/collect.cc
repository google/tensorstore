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

  if (!metric.counters.empty()) {
    for (const auto& v : metric.counters) {
      std::visit(
          [&](auto x) {
            handle_line(
                /*has_value=*/x != 0,
                absl::StrCat(metric_name_with_fields(v), "=", x));
          },
          v.value);
    }
  }
  if (!metric.gauges.empty()) {
    for (const auto& v : metric.gauges) {
      bool has_value = false;
      std::string line = metric_name_with_fields(v);
      std::visit(
          [&](auto x) {
            has_value = (x != 0);
            absl::StrAppend(&line, "={value=", x);
          },
          v.value);
      std::visit(
          [&](auto x) {
            has_value |= (x != 0);
            absl::StrAppend(&line, ", max=", x, "}");
          },
          v.max_value);
      handle_line(has_value, std::move(line));
    }
  }
  if (!metric.histograms.empty()) {
    for (auto& v : metric.histograms) {
      std::string line = metric_name_with_fields(v);
      absl::StrAppend(&line, "={count=", v.count, " sum=", v.sum, " buckets=[");

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
      handle_line(/*has_value=*/v.count || v.sum, std::move(line));
    }
  }
  if (!metric.values.empty()) {
    for (auto& v : metric.values) {
      std::visit(
          [&](auto x) {
            decltype(x) d{};
            handle_line(
                /*has_value=*/x != d,
                absl::StrCat(metric_name_with_fields(v), "=", x));
          },
          v.value);
    }
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
