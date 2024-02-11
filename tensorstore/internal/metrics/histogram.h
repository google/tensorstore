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

#ifndef TENSORSTORE_INTERNAL_METRICS_HISTOGRAM_H_
#define TENSORSTORE_INTERNAL_METRICS_HISTOGRAM_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/debugging/leak_check.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metric_impl.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_metrics {

#ifndef TENSORSTORE_METRICS_DISABLED

/// CounterCell holds an individual "counter" metric value.
template <typename Bucketer>
class HistogramCell;

/// DefaultBucketer buckets by powers of 2:
///  n<0: bucket 0
///  n=0: bucket 1
///  n>0: bucket 1 + log2(n)
struct DefaultBucketer {
  /// Name of Bucketer.
  static constexpr const char kTag[] = "default_histogram";

  /// Number of buckets.
  static constexpr size_t Max = 65;
  static constexpr size_t UnderflowBucket = 0;
  static constexpr size_t OverflowBucket = 64;

  /// Mapping from value to bucket in the range [0 .. Max-1].
  static size_t BucketForValue(double value) {
    static constexpr double kMaximumValue = static_cast<double>(1ull << 63);
    if (value < 0) return UnderflowBucket;
    if (value >= kMaximumValue) return OverflowBucket;
    size_t v = absl::bit_width(static_cast<uint64_t>(value)) + 1;
    return (v >= Max) ? OverflowBucket : v;
  }
};

/// A Histogram metric records a distribution value.
///
/// A Histogram Cell is described by a Bucketer and a set of Fields.
/// The Bucketer maps a value to one of a fixed set of buckets (as in
/// DefaultBucketer). The set of Fields... for each Cell may be int, string, or
/// bool.
///
/// Example:
///   namespace {
///   auto* animals = Histogram<DefaultBucketer, std::string>::New(
///        "/animal/weight", "category");
///   }
///
///   animals->Observe(1.0, "cat");
///   animals->Observe(33.0, "dog");
///
template <typename Bucketer, typename... Fields>
class ABSL_CACHELINE_ALIGNED Histogram {
  using Cell = HistogramCell<Bucketer>;
  using Impl = AbstractMetric<Cell, false, Fields...>;

 public:
  using value_type = double;
  using count_type = int64_t;

  static std::unique_ptr<Histogram> Allocate(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    return absl::WrapUnique(new Histogram(std::string(metric_name),
                                          std::move(metadata),
                                          {std::string(field_names)...}));
  }

  static Histogram& New(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    auto histogram = Allocate(metric_name, field_names..., metadata);
    GetMetricRegistry().Add(histogram.get());
    return *absl::IgnoreLeak(histogram.release());
  }

  auto tag() const { return Cell::kTag; }
  auto metric_name() const { return impl_.metric_name(); }
  const auto& field_names() const { return impl_.field_names(); }
  MetricMetadata metadata() const { return impl_.metadata(); }

  /// Observe a histogram value.
  void Observe(value_type value,
               typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Observe(value);
  }

  value_type GetMean(typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->GetMean() : value_type{};
  }

  count_type GetCount(
      typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->GetCount() : count_type{};
  }

  std::vector<int64_t> GetBucket(
      size_t idx, typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->GetBucket(idx) : std::vector<int64_t>{};
  }

  /// Collect the histogram. There is potential tearing between the sum and the
  /// counts, however the current assumption is that it is unlikely to matter.
  CollectedMetric Collect() const {
    CollectedMetric result{};
    result.tag = Cell::kTag;
    result.metric_name = impl_.metric_name();
    result.metadata = impl_.metadata();
    result.field_names = impl_.field_names_vector();
    impl_.CollectCells([&result](const Cell& cell, const auto& fields) {
      result.histograms.emplace_back(std::apply(
          [&](const auto&... item) {
            std::vector<std::string> fields;
            fields.reserve(sizeof...(item));
            (fields.emplace_back(tensorstore::StrCat(item)), ...);
            return cell.Collect(std::move(fields));
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

  void Reset() { impl_.Reset(); }

 private:
  Histogram(std::string metric_name, MetricMetadata metadata,
            typename Impl::field_names_type field_names)
      : impl_(std::move(metric_name), std::move(metadata),
              std::move(field_names)) {}

  Impl impl_;
};

template <typename Bucketer>
class ABSL_CACHELINE_ALIGNED HistogramCell : public Bucketer {
 public:
  using value_type = double;
  using count_type = int64_t;
  using Bucketer::Max;

  HistogramCell() = default;

  void Observe(double value) {
    if (!std::isfinite(value)) return;  // Ignore INF, NaN.
    size_t idx = Bucketer::BucketForValue(value);
    if (idx < 0 || idx >= Max) return;

    // Use bit0 of count as a spinlock.
    uint64_t count = AcquireCountSpinlock();
    uint64_t new_count = count + 2;

    // Compute a new mean using the method of provisional means.
    double mean = mean_.load(std::memory_order_relaxed);
    double new_mean = mean + (value - mean) / (new_count >> 1);
    mean_.store(new_mean, std::memory_order_relaxed);
    if (new_count > 2) {
      double ssd_delta = (value - mean) * (value - new_mean);
      sum_squared_deviation_.store(
          sum_squared_deviation_.load(std::memory_order_relaxed) + ssd_delta);
    }
    count_ = new_count;  // release spinlock

    buckets_[idx].fetch_add(1, std::memory_order_relaxed);
  }

  // There is potential inconsistency between count/sum/bucket
  double GetMean() const { return mean_.load(std::memory_order_relaxed); }
  int64_t GetCount() const {
    return static_cast<int64_t>(count_.load(std::memory_order_relaxed) >> 1);
  }
  int64_t GetSSD() const {
    return sum_squared_deviation_.load(std::memory_order_relaxed);
  }

  int64_t GetBucket(size_t idx) const {
    if (idx >= Max) return 0;
    return buckets_[idx];
  }

  void Reset() {
    // Use bit0 of count as a spinlock.
    AcquireCountSpinlock();
    mean_.store(0, std::memory_order_relaxed);
    sum_squared_deviation_.store(0, std::memory_order_relaxed);
    for (auto& b : buckets_) {
      b.store(0, std::memory_order_relaxed);
    }
    count_ = 0;  // release spinlock
  }

  CollectedMetric::Histogram Collect(std::vector<std::string> fields) const {
    std::vector<int64_t> buckets;
    buckets.reserve(Bucketer::Max);
    uint64_t count = AcquireCountSpinlock();
    double mean = mean_.load(std::memory_order_relaxed);
    double ssd = sum_squared_deviation_.load(std::memory_order_relaxed);
    count_ = count;  // release spinlock before iterating over buckets
    int64_t bucket_count = 0;
    for (auto& b : buckets_) {
      int64_t x = b.load(std::memory_order_relaxed);
      buckets.push_back(x);
      bucket_count += x;
    }
    return CollectedMetric::Histogram{std::move(fields), bucket_count, mean,
                                      ssd, std::move(buckets)};
  }

 private:
  // Acquires the bit-0 spinlock on count_.
  uint64_t AcquireCountSpinlock() const {
    uint64_t count;
    do {
      count = count_.fetch_or(1);
    } while (count & 1);
    return count;
  }

  mutable std::atomic<uint64_t> count_{0};  // mutable for spinlock.
  std::atomic<double> mean_{0.0};
  std::atomic<double> sum_squared_deviation_{0.0};
  std::array<std::atomic<int64_t>, Max> buckets_{};
};

#else
struct DefaultBucketer;
template <typename Bucketer>
struct HistogramCell {
  using value_type = double;
  static void Observe(value_type value) {}
};

template <typename Bucketer, typename... Fields>
class Histogram {
 public:
  using value_type = double;
  using Cell = HistogramCell<Bucketer>;

  static Histogram& New(
      std::string_view metric_name,
      typename internal::FirstType<std::string_view, Fields>... field_names,
      MetricMetadata metadata) {
    static constexpr Histogram metric;
    return const_cast<Histogram&>(metric);
  }
  static void Observe(value_type value,
                      typename FieldTraits<Fields>::param_type... labels) {}
  static Cell& GetCell(typename FieldTraits<Fields>::param_type... labels) {
    static constexpr Cell cell;
    return const_cast<Cell&>(cell);
  }
};
#endif  // TENSORSTORE_METRICS_DISABLED

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_HISTOGRAM_H_
