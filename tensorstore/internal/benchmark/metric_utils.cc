// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/benchmark/metric_utils.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"

using ::tensorstore::internal_metrics::CollectedMetric;

namespace tensorstore {
namespace internal {

::nlohmann::json CollectMetricsToJson(std::string id, std::string_view prefix) {
  auto json_metrics = ::nlohmann::json::array();

  // Add an identifier for each collection.
  if (!id.empty()) {
    json_metrics.emplace_back(
        ::nlohmann::json{{"name", "/identifier"}, {"values", {id}}});
  }

  auto collected_metrics =
      internal_metrics::GetMetricRegistry().CollectWithPrefix(prefix);
  std::sort(collected_metrics.begin(), collected_metrics.end(),
            [](const CollectedMetric& a, const CollectedMetric& b) {
              return std::tie(a.metric_name, a.field_names) <
                     std::tie(b.metric_name, b.field_names);
            });
  for (auto& metric : collected_metrics) {
    if (internal_metrics::IsCollectedMetricNonZero(metric)) {
      json_metrics.emplace_back(
          internal_metrics::CollectedMetricToJson(metric));
    }
  }

  return json_metrics;
}

::nlohmann::json ReadMetricCollectionFromKvstore(
    const kvstore::Spec& kvstore_spec) {
  ::nlohmann::json metrics;
  if (!kvstore_spec.valid() || kvstore_spec.path.empty() ||
      kvstore_spec.path.back() == '/') {
    return metrics;
  }

  auto kvstore = kvstore::Open(kvstore_spec).result();
  if (!kvstore.ok()) {
    return metrics;
  }

  auto read_metrics = kvstore::Read(kvstore.value(), "").result();

  if (!read_metrics.ok()) {
    return metrics;
  }

  std::string metric_str = std::string(read_metrics.value().value);

  if (metric_str.empty()) {
    return metrics;
  }

  return internal_json::ParseJson(metric_str);
}

absl::Status WriteMetricCollectionToKvstore(::nlohmann::json all_metrics,
                                            const kvstore::Spec& kvstore_spec,
                                            bool final_collect) {
  if (!kvstore_spec.valid() || kvstore_spec.path.empty() ||
      kvstore_spec.path.back() == '/') {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Invalid kvstore_spec");
  }

  if (final_collect) {
    all_metrics.emplace_back(CollectMetricsToJson("final", ""));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto kvstore,
                               kvstore::Open(kvstore_spec).result());

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto write_status,
      kvstore::Write(kvstore, "", absl::Cord(all_metrics.dump())).result());

  std::cout << "Dumped metrics to: " << kvstore_spec.path << std::endl;

  return absl::OkStatus();
}

void DumpMetrics(std::string_view prefix) {
  std::vector<std::string> lines;
  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(prefix)) {
    internal_metrics::FormatCollectedMetric(
        metric, [&lines](bool has_value, std::string line) {
          if (has_value) lines.emplace_back(std::move(line));
        });
  }

  // `lines` is unordered, which isn't great for benchmark comparison.
  std::sort(std::begin(lines), std::end(lines));
  std::cout << std::endl;
  for (const auto& l : lines) {
    std::cout << l << std::endl;
  }
  std::cout << std::endl;
}

};  // namespace internal
};  // namespace tensorstore
