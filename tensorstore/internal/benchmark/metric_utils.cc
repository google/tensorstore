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

#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

::nlohmann::json CollectMetricsToJson(std::string id, std::string_view prefix) {
  auto json_metrics = ::nlohmann::json::array();

  // Add an identifier for each collection.
  if (!id.empty()) {
    json_metrics.emplace_back(
        ::nlohmann::json{{"name", "/identifier"}, {"values", {id}}});
  }

  // collect metrics
  for (auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(prefix)) {
    if (internal_metrics::IsCollectedMetricNonZero(metric)) {
      json_metrics.emplace_back(
          internal_metrics::CollectedMetricToJson(metric));
    }
  }

  return json_metrics;
}

bool WriteMetricCollectionToKvstore(::nlohmann::json all_metrics,
                                    const kvstore::Spec& kvstore_spec) {
  if (!kvstore_spec.valid() || kvstore_spec.path.empty() ||
      kvstore_spec.path.back() == '/') {
    return false;
  }

  all_metrics.emplace_back(CollectMetricsToJson("final", ""));

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto kvstore,
                                  kvstore::Open(kvstore_spec).result());

  TENSORSTORE_CHECK_OK(
      kvstore::Write(kvstore, "", absl::Cord(all_metrics.dump())).result());

  std::cout << "Dumped metrics to: " << kvstore_spec.path << std::endl;

  return true;
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
