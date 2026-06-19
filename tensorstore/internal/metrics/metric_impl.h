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

#include <stddef.h>

#include <array>
#include <atomic>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/node_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/metrics/domain_field.h"  // IWYU pragma: keep
#include "tensorstore/internal/metrics/domain_impl.h"

namespace tensorstore {
namespace internal_metrics {

// Primary: not a domain field.
template <typename Field, typename = void>
struct DomainFieldTraits {
  static constexpr bool kIsDomainField = false;
  static constexpr bool kIsDense = false;
};

// Any type with index()/valid()/kTableSize and MetricIndex ctor.
template <typename Field>
struct DomainFieldTraits<
    Field,
    std::void_t<decltype(std::declval<Field>().index()),
                decltype(std::declval<Field>().valid()),
                decltype(Field::kTableSize), decltype(Field(MetricIndex{0}))>> {
  static constexpr bool kIsDomainField = true;
  static constexpr bool kIsDense = Field::kIsDense;
  using type = Field;
  static constexpr size_t kTableSize = Field::kTableSize;
  static constexpr size_t GetIndex(type f) { return f.index(); }
  static constexpr bool IsValid(type f) { return f.valid(); }
  static constexpr type FromIndex(size_t i) { return type(MetricIndex{i}); }
};

template <>
struct DomainFieldTraits<bool> {
  static constexpr bool kIsDomainField = true;
  static constexpr bool kIsDense = true;
  using type = bool;
  static constexpr size_t kTableSize = 2;
  static constexpr size_t GetIndex(bool field) { return field ? 1 : 0; }
  static constexpr bool IsValid(bool field) { return true; }
  static constexpr bool FromIndex(size_t index) { return index == 1; }
};

template <typename... Ts>
constexpr bool AllDomainFields =
    (sizeof...(Ts) > 0) && (DomainFieldTraits<Ts>::kIsDomainField && ...);

template <typename... Ts>
constexpr bool AllFieldsDense =
    (sizeof...(Ts) > 0) && (DomainFieldTraits<Ts>::kIsDense && ...);

size_t MetricThreadCounter();

template <typename T>
T* LazyInit(std::atomic<T*>& ptr) {
  T* p = ptr.load(std::memory_order_acquire);
  if (ABSL_PREDICT_FALSE(!p)) {
    auto* candidate = absl::IgnoreLeak(new T());
    T* expected = nullptr;
    if (!ptr.compare_exchange_strong(expected, candidate,
                                     std::memory_order_acq_rel)) {
      delete candidate;
      return expected;
    }
    return candidate;
  }
  return p;
}

// Metrics include an optional set of labels of type {int, string, bool}.
template <typename K, typename = void>
struct FieldTraits;

template <>
struct FieldTraits<std::string> {
  using type = std::string;
  using param_type = std::string_view;
  using streamz_type = std::string;
};

template <>
struct FieldTraits<int> {
  using type = int;
  using param_type = int;
  using streamz_type = int;
};

template <>
struct FieldTraits<bool> {
  using type = bool;
  using param_type = bool;
  using streamz_type = bool;
};

template <typename Field>
struct FieldTraits<
    Field, std::void_t<typename Field::T, decltype(Field::kTableSize)>> {
  using type = Field;
  using T = typename Field::T;
  using param_type = Field;
  using streamz_type =
      std::conditional_t<std::is_same_v<T, std::string_view>, std::string, int>;
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

template <typename... Args>
constexpr bool AllValid(const std::tuple<Args...>& t) {
  return std::apply(
      [](const auto&... args) {
        return (
            DomainFieldTraits<std::decay_t<decltype(args)>>::IsValid(args) &&
            ...);
      },
      t);
}

// Indexer converts multiple fields into a flat index array.
template <typename... Fields>
struct Indexer;

template <>
struct Indexer<> {
  static constexpr size_t kSize = 1;
  static constexpr size_t GetIndex() { return 0; }
  static constexpr std::tuple<> GetLabels(size_t index) { return {}; }
};

template <typename FIRST, typename... Rest>
struct Indexer<FIRST, Rest...> {
  using FirstTraits = DomainFieldTraits<FIRST>;
  static constexpr size_t kSize =
      FirstTraits::kTableSize * Indexer<Rest...>::kSize;

  static constexpr size_t GetIndex(FIRST first, Rest... rest) {
    if (!FirstTraits::IsValid(first)) return kInvalidMetricIndex;
    size_t rest_index = Indexer<Rest...>::GetIndex(rest...);
    if (rest_index == kInvalidMetricIndex) return kInvalidMetricIndex;
    return FirstTraits::GetIndex(first) * Indexer<Rest...>::kSize + rest_index;
  }

  static constexpr std::tuple<FIRST, Rest...> GetLabels(size_t index) {
    size_t first_idx = index / Indexer<Rest...>::kSize;
    size_t rest_idx = index % Indexer<Rest...>::kSize;
    return std::tuple_cat(std::make_tuple(FirstTraits::FromIndex(first_idx)),
                          Indexer<Rest...>::GetLabels(rest_idx));
  }
};

// Metric implementation using flat array and perfect hashing.
template <typename Cell, bool HasCombine, typename... Fields>
class PerfectHashMetricImpl {
  using IndexHelper = Indexer<Fields...>;
  static constexpr size_t kTableSize = IndexHelper::kSize;
  static constexpr size_t kAllocSize = kTableSize + 1;

 public:
  using field_values_type = std::tuple<typename FieldTraits<Fields>::type...>;
  using value_type = typename Cell::value_type;
  using CollectCellFn =
      absl::FunctionRef<void(const Cell&, const field_values_type&)>;

  constexpr PerfectHashMetricImpl() : cells_() {}

  ~PerfectHashMetricImpl() = default;

  PerfectHashMetricImpl(const PerfectHashMetricImpl&) = delete;
  PerfectHashMetricImpl& operator=(const PerfectHashMetricImpl&) = delete;

  const Cell* FindCell(
      typename FieldTraits<Fields>::param_type... labels) const {
    size_t idx = IndexHelper::GetIndex(Fields{labels}...);
    if (idx == kInvalidMetricIndex) return nullptr;
    return &cells_[idx];
  }

  Cell* GetCell(typename FieldTraits<Fields>::param_type... labels) const {
    size_t idx = IndexHelper::GetIndex(Fields{labels}...);
    if (idx == kInvalidMetricIndex) {
      ABSL_LOG_FIRST_N(WARNING, 1)
          << "Invalid domain value seen in metric GetCell";
      return &cells_[kTableSize];  // Return overflow cell
    }
    return &cells_[idx];
  }

  bool HasCell(typename FieldTraits<Fields>::param_type... labels) {
    size_t idx = IndexHelper::GetIndex(Fields{labels}...);
    return idx != kInvalidMetricIndex;
  }

  void CollectCells(CollectCellFn on_cell) const {
    for (size_t i = 0; i < kTableSize; ++i) {
      if constexpr (AllFieldsDense<Fields...>) {
        on_cell(cells_[i], IndexHelper::GetLabels(i));
      } else {
        auto labels = IndexHelper::GetLabels(i);
        if (!AllValid(labels)) continue;
        on_cell(cells_[i], labels);
      }
    }
    const auto& overflow = cells_[kTableSize];
    if (!IsDefaultCell(overflow)) {
      on_cell(overflow, GetOverflowLabels());
    }
  }

  void Reset() {
    for (size_t i = 0; i < kAllocSize; ++i) {
      cells_[i].Reset();
    }
  }

 private:
  static constexpr std::tuple<Fields...> GetOverflowLabels() {
    return std::make_tuple(
        DomainFieldTraits<Fields>::FromIndex(kInvalidMetricIndex)...);
  }

  mutable std::array<Cell, kAllocSize> cells_;
};

template <typename Cell, bool HasCombine, typename... Fields>
class AbstractMetric;

template <typename Cell, bool HasCombine, typename... Fields>
struct MetricImplSelectHelper {
  static_assert(AllDomainFields<Fields...> ||
                    (!DomainFieldTraits<Fields>::kIsDomainField && ...),
                "Cannot mix domain fields with regular fields");
  using type =
      std::conditional_t<AllDomainFields<Fields...>,
                         PerfectHashMetricImpl<Cell, HasCombine, Fields...>,
                         AbstractMetric<Cell, HasCombine, Fields...>>;
};

template <typename Cell, bool HasCombine, typename... Fields>
using MetricImplSelect =
    typename MetricImplSelectHelper<Cell, HasCombine, Fields...>::type;

/// AbstractMetric maintains a mapping from a set of field labels to a Cell.
template <typename Cell, bool HasCombine, typename... Fields>
class AbstractMetric {
  using Key = KeyTuple<typename FieldTraits<Fields>::type...>;
  using LookupKey = KeyTuple<typename FieldTraits<Fields>::param_type...>;

  struct State {
    mutable absl::Mutex mu;
    absl::node_hash_map<Key, Cell> map;
  };

 public:
  using field_values_type = typename Key::tuple_type;
  using value_type = typename Cell::value_type;

  constexpr AbstractMetric() : state_(nullptr) {}

  ~AbstractMetric() {
    // NOTE: State* should be "eternal", so we just leak it.
  }

  AbstractMetric(const AbstractMetric&) = delete;
  AbstractMetric& operator=(const AbstractMetric&) = delete;

  const Cell* FindCell(
      typename FieldTraits<Fields>::param_type... labels) const {
    State* state = state_.load(std::memory_order_acquire);
    if (!state) return nullptr;
    LookupKey k{labels...};
    absl::MutexLock l(&state->mu);
    auto it = state->map.find(k);
    return it == state->map.end() ? nullptr : &(it->second);
  }

  Cell* GetCell(typename FieldTraits<Fields>::param_type... labels) const {
    State* state = GetState();
    LookupKey k{labels...};
    absl::MutexLock l(&state->mu);
    return &state->map[std::move(k)];
  }

  bool HasCell(typename FieldTraits<Fields>::param_type... labels) {
    State* state = state_.load(std::memory_order_acquire);
    if (!state) return false;
    LookupKey k{labels...};
    absl::MutexLock l(&state->mu);
    return state->map.count(std::move(k)) > 0;
  }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    State* state = state_.load(std::memory_order_acquire);
    if (!state) return;
    absl::MutexLock l(&state->mu);
    for (auto& kv : state->map) {
      on_cell(kv.second, kv.first.data());
    }
  }

  void Reset() {
    State* state = state_.load(std::memory_order_acquire);
    if (!state) return;
    absl::MutexLock l(&state->mu);
    for (auto& [k, v] : state->map) {
      v.Reset();
    }
  }

 private:
  State* GetState() const { return LazyInit(state_); }

  mutable std::atomic<State*> state_;
};

// Lock-free Specialization for no fields using a sharded counter.
template <typename Cell>
class AbstractMetric<Cell, true> {
 public:
  static constexpr bool kNumThreads = 4;
  using field_values_type = std::tuple<>;
  using value_type = typename Cell::value_type;

  constexpr AbstractMetric() = default;
  ~AbstractMetric() = default;

  AbstractMetric(const AbstractMetric&) = delete;
  AbstractMetric& operator=(const AbstractMetric&) = delete;

  const Cell* FindCell() const { return &cells_[get_id()]; }
  Cell* GetCell() const { return &cells_[get_id()]; }
  bool HasCell() { return true; }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    Cell c;
    for (auto& x : cells_) x.Combine(c);
    field_values_type g;
    on_cell(c, g);
  }

  void Reset() {
    for (auto& x : cells_) x.Reset();
  }

 private:
  size_t get_id() const {
    thread_local size_t id = MetricThreadCounter() & (kNumThreads - 1);
    return id;
  }

  mutable Cell cells_[kNumThreads];
  static_assert(sizeof(cells_) == kNumThreads * ABSL_CACHELINE_SIZE);
};

template <typename Cell, typename Enable = void>
class CellStorage {
 public:
  constexpr CellStorage() : value_() {}
  ~CellStorage() = default;

  Cell* Get() const { return &value_; }
  const Cell* Find() const { return &value_; }
  void Reset() { value_.Reset(); }

 private:
  mutable Cell value_;
};

template <typename Cell>
class CellStorage<Cell,
                  std::enable_if_t<!std::is_trivially_destructible_v<Cell>>> {
 public:
  constexpr CellStorage() : ptr_(nullptr) {}
  ~CellStorage() {
    // NOTE: Cell* should be "eternal", so we just leak it.
  }

  Cell* Get() const { return LazyInit(ptr_); }

  const Cell* Find() const { return ptr_.load(std::memory_order_acquire); }

  void Reset() {
    Cell* p = ptr_.load(std::memory_order_acquire);
    if (p) p->Reset();
  }

 private:
  mutable std::atomic<Cell*> ptr_;
};

// Lock-free Specialization for no fields.
template <typename Cell>
class AbstractMetric<Cell, false> {
 public:
  using field_values_type = std::tuple<>;
  using value_type = typename Cell::value_type;

  constexpr AbstractMetric() = default;
  ~AbstractMetric() = default;

  AbstractMetric(const AbstractMetric&) = delete;
  AbstractMetric& operator=(const AbstractMetric&) = delete;

  const Cell* FindCell() const { return impl_.Find(); }
  Cell* GetCell() const { return impl_.Get(); }
  bool HasCell() { return true; }

  using CollectCellFn = absl::FunctionRef<void(
      const Cell& /*value*/, const field_values_type& /*labels*/)>;

  void CollectCells(CollectCellFn on_cell) const {
    const Cell* p = impl_.Find();
    field_values_type g;
    if (p) {
      on_cell(*p, g);
    } else {
      on_cell(Cell{}, g);
    }
  }

  void Reset() { impl_.Reset(); }

 private:
  CellStorage<Cell> impl_;
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_METRIC_IMPL_H_
