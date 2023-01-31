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
#include <cstdint>
#include <limits>
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
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metric_impl.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_metrics {

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
/// Example:
///   namespace {
///   auto* last_fed = Value<int64_t, absl::Time>::New("/house/last_fed",
///       "name");
///   }
///
///   last_fed->Set("fido", absl::Now());
///
template <typename T, typename... Fields>
class ABSL_CACHELINE_ALIGNED Value {
  using Cell = std::conditional_t<std::is_arithmetic_v<T>, AtomicValueCell<T>,
                                  MutexValueCell<T>>;
  using Impl = AbstractMetric<Cell, Fields...>;

 public:
  using value_type = T;

  static std::unique_ptr<Value> Allocate(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    return absl::WrapUnique(new Value(std::string(metric_name),
                                      std::move(metadata),
                                      {std::string(field_names)...}));
  }

  static Value& New(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    auto value = Allocate(metric_name, field_names..., metadata);
    GetMetricRegistry().Add(value.get());
    return *absl::IgnoreLeak(value.release());
  }

  const auto tag() const { return Cell::kTag; }
  const auto metric_name() const { return impl_.metric_name(); }
  const auto field_names() const { return impl_.field_names(); }
  const MetricMetadata metadata() const { return impl_.metadata(); }

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
  CollectedMetric Collect() const {
    CollectedMetric result{};
    result.tag = Cell::kTag;
    result.metric_name = impl_.metric_name();
    result.metadata = impl_.metadata();
    result.field_names = impl_.field_names_vector();
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.values.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.push_back(tensorstore::StrCat(item)), ...);
            if constexpr (std::is_same_v<Cell, MutexValueCell<T>>) {
              return CollectedMetric::Value{std::move(fields), cell.AsString()};
            } else {
              return CollectedMetric::Value{std::move(fields), cell.Get()};
            }
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
  Value(std::string metric_name, MetricMetadata metadata,
        typename Impl::field_names_type field_names)
      : impl_(std::move(metric_name), std::move(metadata),
              std::move(field_names)) {}

  Impl impl_;
};

struct ValueTag {
  static constexpr const char kTag[] = "value";
};

template <typename T>
class ABSL_CACHELINE_ALIGNED AtomicValueCell : public ValueTag {
 public:
  using value_type = T;
  AtomicValueCell() = default;

  void Set(T value) { value_ = value; }
  T Get() const { return value_; }

 private:
  std::atomic<T> value_{};
};

template <typename T>
class ABSL_CACHELINE_ALIGNED MutexValueCell : public ValueTag {
 public:
  using value_type = T;
  MutexValueCell() = default;

  /// Increment the counter by value.
  void Set(T value) {
    absl::MutexLock l(&m_);
    value_ = std::move(value);
  }
  T Get() const {
    absl::MutexLock l(&m_);
    return value_;
  }

 private:
  friend class Value<T>;

  std::string AsString() const {
    absl::MutexLock l(&m_);
    if constexpr (std::is_same_v<T, std::string>) {
      return value_;
    }
    return tensorstore::StrCat(value_);
  }

  mutable absl::Mutex m_;
  T value_;
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_VALUE_H_
