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

#include <stddef.h>

#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/meta/type_traits.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/fwd.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metric_hook.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/span.h"

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
    bool operator()(CollectMetricTag, CollectedMetric& result) const {
      metric->Collect(result);
      return true;
    }
    void operator()(ResetMetricTag) const { metric->Reset(); }
  };

  using Metric = poly::Poly<sizeof(void*), /*Copyable=*/true,
                            bool(CollectMetricTag, CollectedMetric&) const,
                            void(ResetMetricTag) const>;

  struct GenericMetricWrapper {
    std::function<std::optional<CollectedMetric>()> collect;
    bool operator()(CollectMetricTag, CollectedMetric& result) const {
      auto opt = collect();
      if (opt) {
        result = std::move(*opt);
        return true;
      }
      return false;
    }
    void operator()(ResetMetricTag) const {}
  };

  /// Add a generic metric to be collected. Metric name must be a path-style
  /// string, must be unique, and must ultimately be a string literal.
  void AddGeneric(StaticStringView metric_name,
                  std::function<std::optional<CollectedMetric>()>&& collect,
                  std::shared_ptr<void> hook = nullptr) {
    ABSL_CHECK(IsValidMetricName(metric_name));
    absl::MutexLock lock(mu_);
    RegisterInternalLocked(MetricMetadata(metric_name, ""), "",
                           GenericMetricWrapper{std::move(collect)},
                           std::move(hook));
  }

  template <typename T, typename... Fields>
  void Register(Counter<T, Fields...>* metric, MetricMetadata metadata) {
    RegisterImpl<Counter<T, Fields...>, Fields...>(metric, std::move(metadata));
  }

  template <typename T, typename... Fields>
  void Register(Gauge<T, Fields...>* metric, MetricMetadata metadata) {
    RegisterImpl<Gauge<T, Fields...>, Fields...>(metric, std::move(metadata));
  }

  template <typename T, typename... Fields>
  void Register(MaxGauge<T, Fields...>* metric, MetricMetadata metadata) {
    RegisterImpl<MaxGauge<T, Fields...>, Fields...>(metric,
                                                    std::move(metadata));
  }

  template <typename T, typename... Fields>
  void Register(Value<T, Fields...>* metric, MetricMetadata metadata) {
    RegisterImpl<Value<T, Fields...>, Fields...>(metric, std::move(metadata));
  }

  template <typename Bucketer, typename... Fields>
  void Register(Histogram<Bucketer, Fields...>* metric,
                MetricMetadata metadata) {
    RegisterImpl<Histogram<Bucketer, Fields...>, Fields...>(
        metric, std::move(metadata));
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
    absl::MutexLock lock(mu_);
    collect_hooks_.push_back(std::move(hook));
  }

 private:
  template <typename MetricType, typename... Fields>
  void RegisterImpl(MetricType* metric, MetricMetadata metadata) {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(metadata.metric_name);
    if (it != entries_.end()) {
      ABSL_CHECK(metric != nullptr && it->metric_ptr == metric)
          << "Metric path conflict: " << metadata.metric_name;
      return;
    }
    ABSL_CHECK_EQ(metadata.fields().size(), sizeof...(Fields))
        << "Field count mismatch for metric: " << metadata.metric_name;
    auto field_names_tuple =
        SpanToTuple<typename internal::FirstType<std::string_view, Fields>...>(
            metadata.fields());
    std::shared_ptr<void> hook = RegisterMetricHook(
        metric, metadata.metric_name, metadata, field_names_tuple);
    RegisterInternalLocked(metadata, metric->tag(),
                           CollectableWrapper<MetricType>{metric},
                           std::move(hook), metric);
  }

  void RegisterInternalLocked(MetricMetadata metadata, std::string_view tag,
                              MetricRegistry::Metric m,
                              std::shared_ptr<void> hook = nullptr,
                              const void* metric_ptr = nullptr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  struct Entry {
    MetricMetadata metadata;
    std::string_view tag;
    MetricRegistry::Metric poly;
    std::shared_ptr<void> hook;
    const void* metric_ptr = nullptr;
  };

  struct EntryHash {
    using is_transparent = void;
    size_t operator()(const Entry& entry) const {
      return absl::Hash<std::string_view>{}(entry.metadata.metric_name);
    }
    size_t operator()(std::string_view name) const {
      return absl::Hash<std::string_view>{}(name);
    }
  };

  struct EntryEq {
    using is_transparent = void;
    bool operator()(const Entry& a, const Entry& b) const {
      return a.metadata.metric_name == b.metadata.metric_name;
    }
    bool operator()(const Entry& a, std::string_view b) const {
      return a.metadata.metric_name == b;
    }
    bool operator()(std::string_view a, const Entry& b) const {
      return a == b.metadata.metric_name;
    }
  };

  template <typename Tuple, size_t... I>
  static Tuple SpanToTupleImpl(tensorstore::span<const std::string_view> span,
                               std::index_sequence<I...>) {
    return std::make_tuple(span[I]...);
  }

  template <typename... Args>
  static std::tuple<Args...> SpanToTuple(
      tensorstore::span<const std::string_view> span) {
    return SpanToTupleImpl<std::tuple<Args...>>(
        span, std::make_index_sequence<sizeof...(Args)>{});
  }

  absl::Mutex mu_;
  absl::flat_hash_set<Entry, EntryHash, EntryEq> entries_ ABSL_GUARDED_BY(mu_);
  std::vector<CollectHook> collect_hooks_ ABSL_GUARDED_BY(mu_);
};

/// Returns the global metric registry. Typically this will be the only registry
/// in a process.
MetricRegistry& GetMetricRegistry();

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_REGISTRY_H_
