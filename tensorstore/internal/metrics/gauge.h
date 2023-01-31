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

#ifndef TENSORSTORE_INTERNAL_METRICS_GAUGE_H_
#define TENSORSTORE_INTERNAL_METRICS_GAUGE_H_

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

/// GaugeCell holds an individual gauge metric value.
template <typename T>
class GaugeCell;

/// A Gauge metric represents values that can increase and decrease.
///
/// Gauges are typically used for measured values like temperatures or current
/// memory usage.
///
/// Gauge is parameterized by the type, int64_t or double.
/// Each gauge has one or more Cells, which are described by Fields...,
/// which may be int, string, or bool.
///
/// Example:
///   namespace {
///   auto* temperature = Gauge<double>::New("/my/cpu/temperature");
///   }
///
///   temperature->Set(33.5);
///   temperature->IncrementBy(3.5);
///   temperature->IncrementBy(-3.5);
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Gauge {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
  using Cell = std::conditional_t<std::is_same_v<T, int64_t>,
                                  GaugeCell<int64_t>, GaugeCell<double>>;
  using Impl = AbstractMetric<Cell, Fields...>;

 public:
  using value_type = T;

  static std::unique_ptr<Gauge> Allocate(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    return absl::WrapUnique(new Gauge(std::string(metric_name),
                                      std::move(metadata),
                                      {std::string(field_names)...}));
  }

  static Gauge& New(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    auto gauge = Allocate(metric_name, field_names..., metadata);
    GetMetricRegistry().Add(gauge.get());
    return *absl::IgnoreLeak(gauge.release());
  }

  const auto tag() const { return Cell::kTag; }
  const auto metric_name() const { return impl_.metric_name(); }
  const auto field_names() const { return impl_.field_names(); }
  const MetricMetadata metadata() const { return impl_.metadata(); }

  /// Increment the counter by 1.
  void Increment(typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Increment();
  }

  /// Increment the counter by value .
  void IncrementBy(value_type value,
                   typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->IncrementBy(value);
  }

  /// Decrement the counter by 1.
  void Decrement(typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Decrement();
  }
  /// Decrement the counter by value .
  void DecrementBy(value_type value,
                   typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->DecrementBy(value);
  }

  /// Set the counter to the value.
  void Set(value_type value,
           typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Set(value);
  }

  /// Get the counter.
  value_type Get(typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->Get() : value_type{};
  }

  /// Get the maximum observed counter value.
  value_type GetMax(typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->GetMax() : value_type{};
  }

  /// Collect the gauge.
  CollectedMetric Collect() const {
    CollectedMetric result{};
    result.tag = Cell::kTag;
    result.metric_name = impl_.metric_name();
    result.metadata = impl_.metadata();
    result.field_names = impl_.field_names_vector();
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.gauges.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(tensorstore::StrCat(item)), ...);
            return CollectedMetric::Gauge{std::move(fields), cell.Get(),
                                          cell.GetMax()};
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
  Gauge(std::string metric_name, MetricMetadata metadata,
        typename Impl::field_names_type field_names)
      : impl_(std::move(metric_name), std::move(metadata),
              std::move(field_names)) {}

  Impl impl_;
};

struct GaugeTag {
  static constexpr const char kTag[] = "gauge";
};

template <>
class ABSL_CACHELINE_ALIGNED GaugeCell<double> : public GaugeTag {
 public:
  using value_type = double;
  GaugeCell() = default;

  void IncrementBy(double value) {
    // C++ 20 will add std::atomic::fetch_add support for floating point types
    double old = value_.load(std::memory_order_relaxed);
    while (!value_.compare_exchange_weak(old, old + value)) {
      // repeat
    }
    SetMax(old + value);
  }

  void DecrementBy(int64_t value) { IncrementBy(-value); }
  void Set(double value) {
    value_ = value;
    SetMax(value);
  }

  void Increment() { IncrementBy(1); }
  void Decrement() { DecrementBy(1); }

  double Get() const { return value_; }
  double GetMax() const { return max_; }

 private:
  inline void SetMax(double value) {
    double h = max_.load(std::memory_order_relaxed);
    while (h < value && !max_.compare_exchange_weak(h, value)) {
      // repeat
    }
  }

  std::atomic<double> value_{0};
  std::atomic<double> max_{0};
};

template <>
class ABSL_CACHELINE_ALIGNED GaugeCell<int64_t> : public GaugeTag {
 public:
  using value_type = int64_t;
  GaugeCell() = default;

  /// Increment the counter by value.
  void IncrementBy(int64_t value) {
    int64_t old = value_.fetch_add(value);
    SetMax(old + value);
  }
  void DecrementBy(int64_t value) { IncrementBy(-value); }
  void Set(int64_t value) {
    value_ = value;
    SetMax(value);
  }

  void Increment() { IncrementBy(1); }
  void Decrement() { DecrementBy(1); }

  int64_t Get() const { return value_; }
  int64_t GetMax() const { return max_; }

 private:
  inline void SetMax(int64_t value) {
    int64_t h = max_.load(std::memory_order_relaxed);
    while (h < value && !max_.compare_exchange_weak(h, value)) {
      // repeat
    }
  }

  std::atomic<int64_t> value_{0};
  std::atomic<int64_t> max_{0};
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_GAUGE_H_
