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

#include <cassert>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"

namespace tensorstore {
namespace internal_metrics {

void MetricRegistry::RegisterInternalLocked(MetricMetadata metadata,
                                            std::string_view tag,
                                            MetricRegistry::Metric m,
                                            std::shared_ptr<void> hook,
                                            const void* metric_ptr)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  ABSL_CHECK(m) << metadata.metric_name;
  std::string_view metric_name = metadata.metric_name;
  auto it = entries_.find(metric_name);
  if (it != entries_.end()) {
    ABSL_CHECK(metric_ptr != nullptr && it->metric_ptr == metric_ptr)
        << "Metric path conflict: " << metric_name;
    return;
  }
  entries_.insert(Entry{std::move(metadata), tag, std::move(m), std::move(hook),
                        metric_ptr});
}

std::vector<CollectedMetric> MetricRegistry::CollectWithPrefix(
    std::string_view prefix) {
  std::vector<CollectedMetric> all;
  std::vector<CollectHook> hooks;
  {
    absl::MutexLock l(mu_);
    for (const auto& entry : entries_) {
      if (prefix.empty() ||
          absl::StartsWith(entry.metadata.metric_name, prefix)) {
        CollectedMetric result;
        result.metric_name = entry.metadata.metric_name;
        result.metadata = entry.metadata;
        result.tag = entry.tag;
        for (const auto& name : entry.metadata.fields()) {
          result.field_names.push_back(name);
        }
        if (entry.poly(CollectMetricTag{}, result)) {
          all.emplace_back(std::move(result));
        }
      }
    }
    hooks = collect_hooks_;
  }
  for (auto& hook : hooks) {
    hook(prefix, all);
  }

  return all;
}

std::optional<CollectedMetric> MetricRegistry::Collect(std::string_view name) {
  absl::MutexLock l(mu_);
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  CollectedMetric result;
  result.metric_name = it->metadata.metric_name;
  result.metadata = it->metadata;
  result.tag = it->tag;
  for (const auto& field_name : it->metadata.fields()) {
    result.field_names.push_back(field_name);
  }
  if (it->poly(CollectMetricTag{}, result)) {
    return result;
  }
  return std::nullopt;
}

MetricRegistry& GetMetricRegistry() {
  static absl::NoDestructor<MetricRegistry> registry;
  return *registry;
}

void MetricRegistry::Reset() {
  absl::MutexLock l(mu_);
  for (auto& v : entries_) {
    v.poly(ResetMetricTag{});
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
