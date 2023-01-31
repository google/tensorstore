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

#ifndef TENSORSTORE_INTERNAL_METRICS_COUNTER_H_
#define TENSORSTORE_INTERNAL_METRICS_COUNTER_H_

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/debugging/leak_check.h"
#include "absl/memory/memory.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metric_impl.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_metrics {

/// CounterCell holds an individual "counter" metric value.
template <typename T>
class CounterCell;

/// A Counter metric represents a monotonically increasing value.
/// Do not use a counter to expose a value that can decrease - instead use a
/// Gauge.
///
/// Counter is parameterized by the type, int64_t or double.
/// Each counter has one or more Cells, which are described by Fields...,
/// which may be int, string, or bool.
///
/// Example:
///   namespace {
///   auto* animals = Counter<int64_t, std::string>::New("/house/animals",
///       "category");
///   }
///
///   animals->Increment("cat");
///   animals->Increment("dog");
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Counter {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
  using Cell = std::conditional_t<std::is_same_v<T, int64_t>,
                                  CounterCell<int64_t>, CounterCell<double>>;
  using Impl = AbstractMetric<Cell, Fields...>;

 public:
  using value_type = T;

  static std::unique_ptr<Counter> Allocate(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    return absl::WrapUnique(new Counter(std::string(metric_name),
                                        std::move(metadata),
                                        {std::string(field_names)...}));
  }

  static Counter& New(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    auto counter = Allocate(metric_name, field_names..., metadata);
    GetMetricRegistry().Add(counter.get());
    return *absl::IgnoreLeak(counter.release());
  }

  const auto tag() const { return Cell::kTag; }
  const auto metric_name() const { return impl_.metric_name(); }
  const auto field_names() const { return impl_.field_names(); }
  const MetricMetadata metadata() const { return impl_.metadata(); }

  /// Increment the counter by 1.
  void Increment(typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Increment();
  }

  /// Increment the counter by value (must be > 0).
  void IncrementBy(value_type value,
                   typename FieldTraits<Fields>::param_type... labels) {
    if (value <= value_type{0}) {
      return;
    }
    impl_.GetCell(labels...)->IncrementBy(value);
  }

  value_type Get(typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->Get() : value_type{};
  }

  /// Collect the counter.
  CollectedMetric Collect() const {
    CollectedMetric result{};
    result.tag = Cell::kTag;
    result.metric_name = impl_.metric_name();
    result.metadata = impl_.metadata();
    result.field_names = impl_.field_names_vector();
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.counters.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(tensorstore::StrCat(item)), ...);
            return CollectedMetric::Counter{std::move(fields), cell.Get()};
          },
          fields));
    });
    return result;
  }

  /// Collect the individual Cells: on_cell is invoked for each entry.
  void CollectCells(typename Impl::CollectCellFn on_cell) const {
    return impl_.CollectCells(on_cell);
  }

  /// Expose an individual cell, which avoids frequent lookups.
  Cell& GetCell(typename FieldTraits<Fields>::param_type... labels) {
    return *impl_.GetCell(labels...);
  }

 private:
  Counter(std::string metric_name, MetricMetadata metadata,
          typename Impl::field_names_type field_names)
      : impl_(std::move(metric_name), std::move(metadata),
              std::move(field_names)) {}

  Impl impl_;
};

struct CounterTag {
  static constexpr const char kTag[] = "counter";
};

template <>
class ABSL_CACHELINE_ALIGNED CounterCell<double> : public CounterTag {
 public:
  using value_type = double;
  CounterCell() = default;

  void IncrementBy(double value) {
    if (value <= 0) return;
    // C++ 20 will add std::atomic::fetch_add support for floating point types
    double v = value_.load(std::memory_order_relaxed);
    while (!value_.compare_exchange_weak(v, v + value)) {
      // repeat
    }
  }

  void Increment() { IncrementBy(1); }

  double Get() const { return value_; }

 private:
  std::atomic<double> value_{0.0};
};

template <>
class ABSL_CACHELINE_ALIGNED CounterCell<int64_t> : public CounterTag {
 public:
  using value_type = int64_t;
  CounterCell() = default;

  /// Increment the counter by value.
  void IncrementBy(int64_t value) {
    if (value <= 0) return;
    value_.fetch_add(value);
  }

  void Increment() { IncrementBy(1); }

  int64_t Get() const { return value_; }

 private:
  std::atomic<int64_t> value_{0};
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_COUNTER_H_
