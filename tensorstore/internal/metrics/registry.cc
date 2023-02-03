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
  ABSL_CHECK(m);
  absl::MutexLock l(&mu_);
  ABSL_CHECK(
      entries_.try_emplace(metric_name, Entry{std::move(m), std::move(hook)})
          .second);
}

std::vector<CollectedMetric> MetricRegistry::CollectWithPrefix(
    std::string_view prefix) {
  std::vector<CollectedMetric> all;
  all.reserve(entries_.size());
  absl::MutexLock l(&mu_);
  for (auto& kv : entries_) {
    if (prefix.empty() || absl::StartsWith(kv.first, prefix)) {
      all.emplace_back(kv.second.poly());
      assert(all.back().metric_name == kv.first);
    }
  }
  return all;
}

std::optional<CollectedMetric> MetricRegistry::Collect(std::string_view name) {
  absl::MutexLock l(&mu_);
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  CollectedMetric m = it->second.poly();
  assert(m.metric_name == it->first);
  return m;
}

MetricRegistry& GetMetricRegistry() {
  static internal::NoDestructor<MetricRegistry> registry;
  return *registry;
}

}  // namespace internal_metrics
}  // namespace tensorstore
