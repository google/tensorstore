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

#include "absl/strings/str_join.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_metrics {

void FormatCollectedMetric(
    const CollectedMetric& metric,
    absl::FunctionRef<void(bool has_value, std::string formatted_line)>
        handle_line) {
  std::string field_names;
  if (!metric.field_names.empty()) {
    field_names =
        tensorstore::StrCat("<", absl::StrJoin(metric.field_names, ", "), ">");
  }
  if (!metric.counters.empty()) {
    for (const auto& v : metric.counters) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = tensorstore::StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      std::visit(
          [&](auto x) {
            handle_line(
                /*has_value=*/x != 0,
                tensorstore::StrCat(metric.metric_name, field_names, fields,
                                    "=", x));
          },
          v.value);
    }
  }
  if (!metric.gauges.empty()) {
    for (const auto& v : metric.gauges) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = tensorstore::StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      bool has_value = false;
      std::string value;
      std::visit(
          [&](auto x) {
            has_value = (x != 0);
            value = tensorstore::StrCat("={value=", x);
          },
          v.value);
      std::visit(
          [&](auto x) {
            has_value |= (x != 0);
            absl::StrAppend(&value, ", max=", x, "}");
          },
          v.max_value);
      handle_line(has_value, tensorstore::StrCat(metric.metric_name,
                                                 field_names, fields, value));
    }
  }
  if (!metric.histograms.empty()) {
    for (auto& v : metric.histograms) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = tensorstore::StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      handle_line(
          /*has_value=*/v.count || v.sum,
          tensorstore::StrCat(metric.metric_name, field_names, fields,
                              "={count=", v.count, " sum=", v.sum, " buckets=[",
                              absl::StrJoin(v.buckets, ","), "]}"));
    }
  }
  if (!metric.values.empty()) {
    for (auto& v : metric.values) {
      std::string fields;
      if (!v.fields.empty()) {
        fields = tensorstore::StrCat("[", absl::StrJoin(v.fields, ", "), "]");
      }
      std::visit(
          [&](auto x) {
            decltype(x) d{};
            handle_line(
                /*has_value=*/x != d,
                tensorstore::StrCat(metric.metric_name, field_names, fields,
                                    "=", x));
          },
          v.value);
    }
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
