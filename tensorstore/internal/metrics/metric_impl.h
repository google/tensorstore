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

#ifndef TENSORSTORE_INTERNAL_METRICS_METRIC_IMPL_H_
#define TENSORSTORE_INTERNAL_METRICS_METRIC_IMPL_H_

// Implementation details for internal/metrics
// IWYU pragma: private

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_metrics {

// Metrics include an optional set of labels of type {int, string, bool}.
template <typename K>
struct FieldTraits;

template <>
struct FieldTraits<std::string> {
  using type = std::string;
  using param_type = std::string_view;
};

template <>
struct FieldTraits<int> {
  using type = int;
  using param_type = int;
};

template <>
struct FieldTraits<bool> {
  using type = bool;
  using param_type = bool;
};

// Metrics allow up to 3 fields.
template <size_t N>
struct FieldNamesTuple;

template <>
struct FieldNamesTuple<0> {
  using type = std::tuple<>;
};
template <>
struct FieldNamesTuple<1> {
  using type = std::tuple<std::string>;
};
template <>
struct FieldNamesTuple<2> {
  using type = std::tuple<std::string, std::string>;
};
template <>
struct FieldNamesTuple<3> {
  using type = std::tuple<std::string, std::string, std::string>;
};

/// An immutable std::tuple<Ts...> with a cached hash.
template <typename... Ts>
class KeyTuple {
 public:
  using tuple_type = std::tuple<Ts...>;

  explicit KeyTuple(Ts... ts)
      : data_(std::move(ts)...),
        hash_(absl::Hash<std::tuple<Ts...>>()(data_)) {}

  template <typename... Us>
  KeyTuple(const KeyTuple<Us...>& other)
      : data_(other.data()), hash_(other.hash()) {}

  template <typename... Us>
  KeyTuple(KeyTuple<Us...>&& other)
      : data_(other.data()), hash_(other.hash()) {}

  size_t hash() const { return hash_; }
  const tuple_type& data() const { return data_; }

  template <typename... Us>
  friend bool operator==(const KeyTuple& a, const KeyTuple<Us...>& b) {
    return a.hash() == b.hash() && a.data() == b.data();
  }

  template <typename... Us>
  friend bool operator!=(const KeyTuple& a, const KeyTuple<Us...>& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const KeyTuple& t) {
    return H::combine(std::move(h), t.hash());
  }

 private:
  tuple_type data_;
  size_t hash_;
};

/// AbstractMetricBase holds the common metric fields for all metric
/// specializations.
template <size_t FieldCount>
class AbstractMetricBase {
 public:
  using field_names_type = typename FieldNamesTuple<FieldCount>::type;

  AbstractMetricBase(std::string metric_name, MetricMetadata metadata,
                     field_names_type field_names)
      : metric_name_(std::move(metric_name)),
        metadata_(std::move(metadata)),
        field_names_(std::move(field_names)) {
    ABSL_CHECK(IsValidMetricName(metric_name_));
    ABSL_CHECK(std::apply(
        [&](const auto&... item) {
          return true && (IsValidMetricLabel(item) && ...);
        },
        field_names_));
  }

  ~AbstractMetricBase() = default;

  AbstractMetricBase(const AbstractMetricBase&) = delete;
  AbstractMetricBase& operator=(const AbstractMetricBase&) = delete;

  const std::string_view metric_name() const { return metric_name_; }
  const MetricMetadata metadata() const { return metadata_; }
  const field_names_type& field_names() const { return field_names_; }

  std::vector<std::string_view> field_names_vector() const {
    if constexpr (FieldCount == 0) {
      return {};
    }
    return std::apply(
        [&](const auto&... item) {
          std::vector<std::string_view> result;
          result.reserve(sizeof...(item));
          (result.emplace_back(std::string_view(item)), ...);
          return result;
        },
        field_names_);
  }

 private:
  std::string metric_name_;
  MetricMetadata metadata_;
  field_names_type field_names_;
};

/// AbstractMetric maintains a mapping from a set of field labels to a Cell.
template <typename Cell, typename... Fields>
class AbstractMetric : public AbstractMetricBase<sizeof...(Fields)> {
  using Base = AbstractMetricBase<sizeof...(Fields)>;

  using Key = KeyTuple<typename FieldTraits<Fields>::type...>;
  using LookupKey = KeyTuple<typename FieldTraits<Fields>::param_type...>;

 public:
  using field_names_type = typename Base::field_names_type;
  using field_values_type = typename Key::tuple_type;
  using value_type = typename Cell::value_type;

  ~AbstractMetric() = default;

  using Base::Base;
  using Base::field_names;
  using Base::field_names_vector;
  using Base::metadata;
  using Base::metric_name;

  const Cell* FindCell(
      typename FieldTraits<Fields>::param_type... labels) const {
    LookupKey k{labels...};
    absl::MutexLock l(&mu_);
    auto it = impl_.find(k);
    return it == impl_.end() ? nullptr : &(it->second);
  }

  Cell* GetCell(typename FieldTraits<Fields>::param_type... labels) {
    LookupKey k{labels...};
    absl::MutexLock l(&mu_);
    return &impl_[std::move(k)];
  }

  bool HasCell(typename FieldTraits<Fields>::param_type... labels) {
    LookupKey k{labels...};
    absl::MutexLock l(&mu_);
    return impl_.count(std::move(k)) > 0;
  }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    absl::MutexLock l(&mu_);
    for (auto& kv : impl_) {
      on_cell(kv.second, kv.first.data());
    }
  }

 private:
  // NOTE: It would be nice to have a lock-free hashtable here.
  mutable absl::Mutex mu_;
  absl::node_hash_map<Key, Cell> impl_;
};

// Lock-free Specialization for no fields.
template <typename Cell>
class AbstractMetric<Cell> : public AbstractMetricBase<0> {
  using Base = AbstractMetricBase<0>;

 public:
  using field_names_type = typename Base::field_names_type;
  using field_values_type = std::tuple<>;
  using value_type = typename Cell::value_type;

  ~AbstractMetric() = default;

  using Base::Base;
  using Base::field_names;
  using Base::field_names_vector;
  using Base::metadata;
  using Base::metric_name;

  const Cell* FindCell() const { return &impl_; }
  Cell* GetCell() { return &impl_; }
  bool HasCell() { return true; }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    field_values_type g;
    on_cell(impl_, g);
  }

 private:
  Cell impl_;
};

// Lock-free Specialization for a single boolean field.
template <typename Cell>
class AbstractMetric<Cell, bool> : public AbstractMetricBase<1> {
  using Base = AbstractMetricBase<1>;

 public:
  using field_names_type = typename Base::field_names_type;
  using field_values_type = std::tuple<bool>;
  using value_type = typename Cell::value_type;

  ~AbstractMetric() = default;

  using Base::Base;
  using Base::field_names;
  using Base::field_names_vector;
  using Base::metadata;
  using Base::metric_name;

  const Cell* FindCell(bool x) const { return x ? &true_impl_ : &false_impl_; }
  Cell* GetCell(bool x) { return x ? &true_impl_ : &false_impl_; }
  bool HasCell(bool) { return true; }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    on_cell(true_impl_, field_values_type{true});
    on_cell(false_impl_, field_values_type{false});
  }

 private:
  Cell true_impl_;
  Cell false_impl_;
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_METRIC_IMPL_H_
