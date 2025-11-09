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

#ifndef TENSORSTORE_INTERNAL_METRICS_REGISTRY_H_
#define TENSORSTORE_INTERNAL_METRICS_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metric_hook.h"
#include "tensorstore/internal/poly/poly.h"

namespace tensorstore {
namespace internal_metrics {

/// Registry which tracks metrics, similar to a Prometheus registry.
/// Metrics can be queried individually or based on a prefix. Note
/// that collection happens under lock, limiting collection parallelism.
class MetricRegistry {
 public:
  struct CollectMetricTag {};
  struct ResetMetricTag {};

  template <typename T>
  struct CollectableWrapper {
    T* metric;
    std::optional<CollectedMetric> operator()(CollectMetricTag) const {
      return metric->Collect();
    }
    void operator()(ResetMetricTag) { metric->Reset(); }
  };

  using Metric =
      poly::Poly<sizeof(void*), /*Copyable=*/true,
                 std::optional<CollectedMetric>(CollectMetricTag) const,
                 void(ResetMetricTag)>;

  /// Add a generic metric to be collected. Metric name must be a path-style
  /// string, must be unique, and must ultimately be a string literal.  This
  /// metric doesn't support reset.
  /// TODO(ChromeHearts) - add support for custom Reset()
  void AddGeneric(std::string_view metric_name,
                  std::function<std::optional<CollectedMetric>()>&& collect,
                  std::shared_ptr<void> hook = nullptr) {
    ABSL_CHECK(IsValidMetricName(metric_name));
    AddInternal(
        metric_name,
        [collect = std::move(collect)](auto p) {
          if constexpr (std::is_same_v<decltype(p), CollectMetricTag>)
            return collect();
          // ignore  ResetMetricTag
        },
        std::move(hook));
  }

  /// Add a metric which is collectable via metric->Collect()
  /// Metric name must be a path-style string, and must be unique.
  template <typename Collectable>
  void Add(Collectable* metric) {
    std::shared_ptr<void> hook = RegisterMetricHook(metric);
    AddInternal(metric->metric_name(), CollectableWrapper<Collectable>{metric},
                std::move(hook));
  }

  /// Collect an individual metric.
  std::optional<CollectedMetric> Collect(std::string_view name);

  /// Collect metrics that begin with the specified prefix.
  /// The result is not ordered.
  std::vector<CollectedMetric> CollectWithPrefix(std::string_view prefix);

  // Reset all the metrics in the registry
  void Reset();

  /// invoked with a metric prefix
  using CollectHook =
      std::function<void(std::string_view, std::vector<CollectedMetric>&)>;

  /// A CollectHook is called post-collection with the prefix.
  void AddCollectHook(CollectHook&& hook) {
    absl::MutexLock lock(&mu_);
    collect_hooks_.push_back(std::move(hook));
  }

 private:
  void AddInternal(std::string_view metric_name, MetricRegistry::Metric m,
                   std::shared_ptr<void> hook = nullptr);

  struct Entry {
    MetricRegistry::Metric poly;
    std::shared_ptr<void> hook;
  };
  absl::Mutex mu_;
  absl::flat_hash_map<std::string_view, Entry> entries_;
  std::vector<CollectHook> collect_hooks_ ABSL_GUARDED_BY(mu_);
};

/// Returns the global metric registry. Typically this will be the only registry
/// in a process.
MetricRegistry& GetMetricRegistry();

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_REGISTRY_H_
