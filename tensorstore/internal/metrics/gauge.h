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

#include <stdint.h>

#include <atomic>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metric_impl.h"

namespace tensorstore {
namespace internal_metrics {

#ifndef TENSORSTORE_METRICS_DISABLED

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
///   TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
///       temperature, Gauge<double>,
///       MetricMetadata("/my/cpu/temperature", "CPU temperature"));
///
///   temperature.Set(33.5);
///   temperature.IncrementBy(3.5);
///   temperature.IncrementBy(-3.5);
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Gauge {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
  using Cell = GaugeCell<T>;
  using Impl = MetricImplSelect<Cell, false, Fields...>;

 public:
  using value_type = T;

  constexpr Gauge() = default;

  Gauge(const Gauge&) = delete;
  Gauge& operator=(const Gauge&) = delete;

  static constexpr std::string_view tag() { return Cell::kTag; }

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
  void Collect(CollectedMetric& result) const {
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.values.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(absl::StrCat(item)), ...);
            return CollectedMetric::Value{std::move(fields), cell.Get(),
                                          cell.GetMax()};
          },
          fields));
    });
  }

  /// Collect the individual Cells: on_cell is invoked for each entry.
  void CollectCells(typename Impl::CollectCellFn on_cell) const {
    return impl_.CollectCells(on_cell);
  }

  /// Expose an individual cell, which avoids frequent lookups.
  Cell& GetCell(typename FieldTraits<Fields>::param_type... labels) const {
    return *impl_.GetCell(labels...);
  }

  void Reset() { impl_.Reset(); }

 private:
  Impl impl_;
};


// GaugeCell
struct GaugeTag {
  static constexpr const char kTag[] = "gauge";
};

template <>
class ABSL_CACHELINE_ALIGNED GaugeCell<double> : public GaugeTag {
 public:
  using value_type = double;
  constexpr GaugeCell() : value_(0.0), max_(0.0) {}

  void IncrementBy(double value) {
    // C++ 20 will add std::atomic::fetch_add support for floating point types
    double old = value_.load(std::memory_order_relaxed);
    while (!value_.compare_exchange_weak(old, old + value)) {
      // repeat
    }
    SetMax(old + value);
  }

  void DecrementBy(double value) { IncrementBy(-value); }
  void Set(double value) {
    value_ = value;
    SetMax(value);
  }

  void Increment() { IncrementBy(1); }
  void Decrement() { DecrementBy(1); }

  double Get() const { return value_; }
  double GetMax() const { return max_; }

  void Reset() {
    // not thread safe
    value_ = 0.0;
    max_ = 0.0;
    SetMax(value_.load());
  }

 private:
  void SetMax(double value) {
    double h = max_.load(std::memory_order_relaxed);
    while (h < value && !max_.compare_exchange_weak(h, value)) {
      // repeat
    }
  }

  std::atomic<double> value_;
  std::atomic<double> max_;
};

template <>
class ABSL_CACHELINE_ALIGNED GaugeCell<int64_t> : public GaugeTag {
 public:
  using value_type = int64_t;
  constexpr GaugeCell() : value_(0), max_(0) {}

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

  void Reset() {
    // not thread safe
    value_ = 0;
    max_ = 0;
    SetMax(value_.load());
  }

 private:
  void SetMax(int64_t value) {
    int64_t h = max_.load(std::memory_order_relaxed);
    while (h < value && !max_.compare_exchange_weak(h, value)) {
      // repeat
    }
  }

  std::atomic<int64_t> value_;
  std::atomic<int64_t> max_;
};


#else
template <typename T>
struct GaugeCell {
  static void IncrementBy(T value) {}

  static void DecrementBy(T value) {}
  static void Set(T value) {}

  static void Increment() { IncrementBy(1); }
  static void Decrement() { DecrementBy(1); }

  static T Get() { return {}; }
  static T GetMax() { return {}; }

  static void Reset() {}
};
template <typename T, typename... Fields>
class Gauge {
 public:
  using value_type = T;
  using Cell = GaugeCell<T>;

  constexpr Gauge() = default;

  static constexpr std::string_view tag() { return "gauge"; }
  static void Increment(typename FieldTraits<Fields>::param_type... labels) {}
  static void IncrementBy(value_type value,
                          typename FieldTraits<Fields>::param_type... labels) {}
  static void Decrement(typename FieldTraits<Fields>::param_type... labels) {}
  static void DecrementBy(value_type value,
                          typename FieldTraits<Fields>::param_type... labels) {}
  static void Set(value_type value,
                  typename FieldTraits<Fields>::param_type... labels) {}
  static Cell& GetCell(typename FieldTraits<Fields>::param_type... labels) {
    static constexpr Cell cell;
    return const_cast<Cell&>(cell);
  }
};

#endif  // TENSORSTORE_METRICS_DISABLED

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_GAUGE_H_
