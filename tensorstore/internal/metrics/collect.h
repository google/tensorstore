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

#ifndef TENSORSTORE_INTERNAL_METRICS_COLLECT_H_
#define TENSORSTORE_INTERNAL_METRICS_COLLECT_H_

#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "tensorstore/internal/metrics/metadata.h"

namespace tensorstore {
namespace internal_metrics {

/// CollectedMetric contains the data of a given metric at a point in time.
struct CollectedMetric {
  std::string_view metric_name;
  std::vector<std::string_view> field_names;
  MetricMetadata metadata;
  std::string_view tag;

  struct Counter {
    std::vector<std::string> fields;
    std::variant<int64_t, double> value;
  };
  std::vector<Counter> counters;

  struct Gauge {
    std::vector<std::string> fields;
    std::variant<int64_t, double> value;
    std::variant<int64_t, double> max_value;
  };
  std::vector<Gauge> gauges;

  struct Histogram {
    std::vector<std::string> fields;
    int64_t count;
    double sum;
    std::vector<int64_t> buckets;
  };
  std::vector<Histogram> histograms;

  struct Value {
    std::vector<std::string> fields;
    std::variant<int64_t, double, std::string> value;
  };
  std::vector<Value> values;
};

/// Invokes handle_line on one or more formatted CollectedMetric lines.
void FormatCollectedMetric(
    const CollectedMetric& metric,
    absl::FunctionRef<void(bool has_value, std::string formatted_line)>
        handle_line);

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_COLLECT_H_
