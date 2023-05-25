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

#include "tensorstore/internal/metrics/registry.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/no_destructor.h"

namespace tensorstore {
namespace internal_metrics {

void MetricRegistry::AddInternal(std::string_view metric_name,
                                 MetricRegistry::Metric m,
                                 std::shared_ptr<void> hook) {
  ABSL_CHECK(m) << metric_name;
  absl::MutexLock l(&mu_);
  ABSL_CHECK(
      entries_.try_emplace(metric_name, Entry{std::move(m), std::move(hook)})
          .second)
      << metric_name;
}

std::vector<CollectedMetric> MetricRegistry::CollectWithPrefix(
    std::string_view prefix) {
  std::vector<CollectedMetric> all;
  all.reserve(entries_.size());
  absl::MutexLock l(&mu_);
  for (auto& kv : entries_) {
    if (prefix.empty() || absl::StartsWith(kv.first, prefix)) {
      auto opt_metric = kv.second.poly(CollectMetricTag{});
      if (opt_metric.has_value()) {
        all.emplace_back(std::move(*opt_metric));
        assert(all.back().metric_name == kv.first);
      }
    }
  }
  for (auto& hook : collect_hooks_) {
    hook(prefix, all);
  }

  return all;
}

std::optional<CollectedMetric> MetricRegistry::Collect(std::string_view name) {
  absl::MutexLock l(&mu_);
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  auto opt_metric = it->second.poly(CollectMetricTag{});
  assert(!opt_metric.has_value() || opt_metric->metric_name == it->first);
  return opt_metric;
}

MetricRegistry& GetMetricRegistry() {
  static internal::NoDestructor<MetricRegistry> registry;
  return *registry;
}

void MetricRegistry::Reset() {
  absl::MutexLock l(&mu_);
  for (auto& [k, v] : entries_) {
    v.poly(ResetMetricTag{});
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
