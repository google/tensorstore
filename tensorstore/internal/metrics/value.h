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

#ifndef TENSORSTORE_INTERNAL_METRICS_VALUE_H_
#define TENSORSTORE_INTERNAL_METRICS_VALUE_H_

#include <atomic>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metric_impl.h"

namespace tensorstore {
namespace internal_metrics {

#ifndef TENSORSTORE_METRICS_DISABLED

/// ValueCell holds an individual "value" metric value.
template <typename T>
class AtomicValueCell;
template <typename T>
class MutexValueCell;

/// A Value metric represents a point value.
///
/// Value is parameterized by the type, which is a signed int (int64_t),
/// floating point (double), or a value convertible to a string.
/// Each value has one or more Cells, which are described by Fields...,
/// which may be int, string, or bool.
///
///   TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
///       last_fed, (Value<int64_t, std::string>),
///       MetricMetadata("/house/last_fed", "Last time fed"), "name");
///
///   last_fed.Set(absl::ToUnixMillis(absl::Now()), "fido");
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Value {
  using Cell = std::conditional_t<std::is_arithmetic_v<T>, AtomicValueCell<T>,
                                  MutexValueCell<T>>;
  using Impl = MetricImplSelect<Cell, false, Fields...>;

 public:
  using value_type = T;

  constexpr Value() = default;

  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  static constexpr std::string_view tag() { return Cell::kTag; }

  /// Set the value.
  void Set(value_type value,
           typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Set(value);
  }

  /// Get the value.
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
            if constexpr (std::is_same_v<Cell, MutexValueCell<T>>) {
              return CollectedMetric::Value{std::move(fields), cell.AsString()};
            } else {
              return CollectedMetric::Value{std::move(fields), cell.Get()};
            }
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

struct ValueTag {
  static constexpr const char kTag[] = "value";
};

template <typename T>
class ABSL_CACHELINE_ALIGNED AtomicValueCell : public ValueTag {
 public:
  using value_type = T;
  constexpr AtomicValueCell() : value_() {}

  void Set(T value) { value_ = value; }
  T Get() const { return value_; }

  void Reset() { Set(T()); }

 private:
  std::atomic<T> value_;
};

template <typename T>
class ABSL_CACHELINE_ALIGNED MutexValueCell : public ValueTag {
 public:
  using value_type = T;
  constexpr MutexValueCell() : value_() {}

  /// Increment the counter by value.
  void Set(T value) {
    absl::MutexLock l(m_);
    value_ = std::move(value);
  }
  T Get() const {
    absl::MutexLock l(m_);
    return value_;
  }
  void Reset() { Set(T()); }

 private:
  friend class Value<T>;

  std::string AsString() const {
    absl::MutexLock l(m_);
    if constexpr (std::is_same_v<T, std::string>) {
      return value_;
    }
    return absl::StrCat(value_);
  }

  mutable absl::Mutex m_;
  T value_;
};

#else
template <typename T, typename... Fields>
class Value {
 public:
  using value_type = T;

  constexpr Value() = default;

  static constexpr std::string_view tag() { return "value"; }
  static void Set(value_type value,
                  typename FieldTraits<Fields>::param_type... labels) {}
};
#endif  // TENSORSTORE_METRICS_DISABLED

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_VALUE_H_
