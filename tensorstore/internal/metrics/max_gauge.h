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

#ifndef TENSORSTORE_INTERNAL_METRICS_MAX_GAUGE_H_
#define TENSORSTORE_INTERNAL_METRICS_MAX_GAUGE_H_

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
class MaxCell;

/// A MaxGauge metric tracks the max of a value.
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED MaxGauge {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double>);
  using Cell = MaxCell<T>;
  using Impl = MetricImplSelect<Cell, true, Fields...>;

 public:
  using value_type = T;

  constexpr MaxGauge() = default;

  MaxGauge(const MaxGauge&) = delete;
  MaxGauge& operator=(const MaxGauge&) = delete;

  static constexpr std::string_view tag() { return Cell::kTag; }

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

  /// Collect the gauge.
  void Collect(CollectedMetric& result) const {
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.values.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(absl::StrCat(item)), ...);
            return CollectedMetric::Value{std::move(fields), {}, cell.Get()};
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

template <typename T>
class ABSL_CACHELINE_ALIGNED MaxCell {
 public:
  using value_type = T;
  static constexpr const char kTag[] = "max_gauge";

  constexpr MaxCell() : max_(0) {}

  void Set(value_type value) {
    value_type h = max_.load(std::memory_order_relaxed);
    while (h < value && !max_.compare_exchange_weak(h, value)) {
      // repeat
    }
  }

  value_type Get() const { return max_; }

  void Reset() { max_ = 0; }

  void Combine(MaxCell& other) const {
    other.Set(max_.load(std::memory_order_relaxed));
  }

 private:
  std::atomic<T> max_;
};

#else

template <typename T, typename... Fields>
class MaxGauge : public Gauge<T, Fields...> {
 public:
  using value_type = T;

  constexpr MaxGauge() = default;

  static constexpr std::string_view tag() { return "max_gauge"; }
};

#endif  // TENSORSTORE_METRICS_DISABLED

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_MAX_GAUGE_H_
