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

#include <array>
#include <atomic>
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

  /// Mapping from value to bucket in the range [0 .. Max-1].
  static size_t BucketForValue(double value) {
    static constexpr double kTop = static_cast<double>(1ull << 63);
    if (value >= kTop) return Max - 1;                 // Inf.
    if (value < 0 || !std::isfinite(value)) return 0;  // <0, NaN
    size_t v = absl::bit_width(static_cast<uint64_t>(value)) + 1;
    return (v >= Max) ? (Max - 1) : v;
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
  using Impl = AbstractMetric<Cell, Fields...>;

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

  const auto tag() const { return Cell::kTag; }
  const auto metric_name() const { return impl_.metric_name(); }
  const auto field_names() const { return impl_.field_names(); }
  const MetricMetadata metadata() const { return impl_.metadata(); }

  /// Observe a histogram value.
  void Observe(value_type value,
               typename FieldTraits<Fields>::param_type... labels) {
    impl_.GetCell(labels...)->Observe(value);
  }

  value_type GetSum(typename FieldTraits<Fields>::param_type... labels) const {
    auto* cell = impl_.FindCell(labels...);
    return cell ? cell->GetSum() : value_type{};
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

            int64_t count = 0;
            std::vector<int64_t> buckets;
            buckets.reserve(cell.Max);
            for (size_t i = 0; i < cell.Max; i++) {
              buckets.push_back(cell.GetBucket(i));
              count += buckets.back();
            }
            return CollectedMetric::Histogram{
                std::move(fields), count, cell.GetSum(), std::move(buckets)};
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
    size_t idx = Bucketer::BucketForValue(value);
    if (idx < 0 || idx >= Max) return;

    double v = sum_.load();
    while (!sum_.compare_exchange_weak(v, v + value)) {
      // repeat
    }
    count_.fetch_add(1);
    buckets_[idx].fetch_add(1);
  }

  // There is potential inconsistency between count/sum/bucket
  double GetSum() const { return sum_; }
  int64_t GetCount() const { return count_; }
  int64_t GetBucket(size_t idx) const {
    if (idx >= Max) return 0;
    return buckets_[idx];
  }

 private:
  std::atomic<double> sum_{0.0};
  std::atomic<int64_t> count_{0};
  std::array<std::atomic<int64_t>, Max> buckets_{};
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_HISTOGRAM_H_
