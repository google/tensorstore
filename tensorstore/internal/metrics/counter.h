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
///   TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
///       animals, (Counter<int64_t, std::string>),
///       MetricMetadata("/house/animals", "House animals count"), "category");
///
///   animals.Increment("cat");
///   animals.Increment("dog");
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Counter {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
  using Cell = std::conditional_t<std::is_same_v<T, int64_t>,
                                  CounterCell<int64_t>, CounterCell<double>>;
  using Impl = MetricImplSelect<Cell, true, Fields...>;

 public:
  using value_type = T;

  constexpr Counter() = default;

  Counter(const Counter&) = delete;
  Counter& operator=(const Counter&) = delete;

  static constexpr std::string_view tag() { return Cell::kTag; }

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
  void Collect(CollectedMetric& result) const {
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.values.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(absl::StrCat(item)), ...);
            return CollectedMetric::Value{std::move(fields), cell.Get()};
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

struct CounterTag {
  static constexpr const char kTag[] = "counter";
};

template <>
class ABSL_CACHELINE_ALIGNED CounterCell<double> : public CounterTag {
 public:
  using value_type = double;
  constexpr CounterCell() : value_(0.0) {}

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

  void Reset() { value_ = 0.0; }

  void Combine(CounterCell& other) const {
    other.value_.store(other.value_.load(std::memory_order_relaxed) +
                       value_.load(std::memory_order_relaxed));
  }

 private:
  std::atomic<double> value_;
};

template <>
class ABSL_CACHELINE_ALIGNED CounterCell<int64_t> : public CounterTag {
 public:
  using value_type = int64_t;
  constexpr CounterCell() : value_(0) {}

  /// Increment the counter by value.
  void IncrementBy(int64_t value) {
    if (value <= 0) return;
    value_.fetch_add(value);
  }

  void Increment() { IncrementBy(1); }

  int64_t Get() const { return value_; }

  void Reset() { value_ = 0; }

  void Combine(CounterCell& other) const {
    other.value_.fetch_add(value_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
  }

 private:
  std::atomic<int64_t> value_;
};

#else
template <typename T>
struct CounterCell {
  static void IncrementBy(T value) {}
  static void Increment() { IncrementBy(1); }
};
template <typename T, typename... Fields>
class Counter {
 public:
  using value_type = T;
  using Cell = CounterCell<T>;

  constexpr Counter() = default;

  static constexpr std::string_view tag() { return "counter"; }
  static void Increment(typename FieldTraits<Fields>::param_type... labels) {}
  static void IncrementBy(value_type value,
                          typename FieldTraits<Fields>::param_type... labels) {}
  static Cell& GetCell(typename FieldTraits<Fields>::param_type... labels) {
    static constexpr Cell cell;
    return const_cast<Cell&>(cell);
  }
};
#endif  // TENSORSTORE_METRICS_DISABLED

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_COUNTER_H_
