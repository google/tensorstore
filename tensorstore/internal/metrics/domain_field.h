// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_METRICS_DOMAIN_FIELD_H_
#define TENSORSTORE_INTERNAL_METRICS_DOMAIN_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string_view>
#include <type_traits>

#include "absl/strings/str_format.h"
#include "tensorstore/internal/metrics/domain_impl.h"

namespace tensorstore {
namespace internal_metrics {

// Indexing type to avoid accidental implicit conversion to an index type.
struct MetricIndex {
  size_t value;
  explicit constexpr MetricIndex(size_t v) : value(v) {}
};

// Wrapper for a metric field with a known, static domain.
//
// Supports `std::string_view`, integral, and enum types.
//
// If the domain is dense-sequential (for integral and enum types), maps values
// directly to indices via `value - Min`, avoiding hashing entirely.
// Otherwise, uses compile-time perfect hashing (via FNV-1a and a precomputed or
// searched seed) to map values to a dense index range `[0, kTableSize - 1]`.
//
// Invalid values (not present in the domain) are mapped to an invalid index
// (kInvalidIndex), and stringify to "".
template <typename Spec, bool CaseSensitive = false>
class DomainField {
 public:
  using T = typename decltype(Spec::kValues)::value_type;
  using UnderlyingT = UnderlyingTypeOrSelfT<T>;

  static_assert(std::is_integral_v<T> || std::is_enum_v<T> ||
                    std::is_same_v<T, std::string_view>,
                "DomainField type must be integral, enum, or string_view");

  using SpecType = Spec;
  static constexpr auto& kValues = Spec::kValues;
  static constexpr size_t kSize = kValues.size();
  static_assert(kSize > 0, "Domain cannot be empty");

  static constexpr bool kCaseSensitive =
      !(std::is_integral_v<T> || std::is_enum_v<T>) &&
      GetCaseSensitiveV<Spec, CaseSensitive>;

  static_assert(!HasDuplicates<T, kSize, kCaseSensitive>(kValues),
                "Domain has duplicate values");

  static constexpr bool kIsDense = IsDenseSequential<T, kSize>(kValues);

  static_assert(kIsDense || HasTableSizeV<Spec>,
                "Spec is missing kTableSize. Use find_seed tool.");
  static_assert(kIsDense || HasSeedV<Spec>,
                "Spec is missing kSeed. Use find_seed tool.");

  static constexpr T kMin = FindMin(kValues);
  static constexpr T kMax = FindMax(kValues);

  static constexpr size_t kTableSize = [] {
    if constexpr (kIsDense) {
      return kSize;
    } else {
      return Spec::kTableSize;
    }
  }();

  static constexpr uint32_t kSeed = [] {
    if constexpr (kIsDense) {
      return 0;
    } else {
      return Spec::kSeed;
    }
  }();

  static_assert(kIsDense ||
                    IsPerfect<T, kSize, kCaseSensitive, kTableSize>(kValues,
                                                                    kSeed),
                "Provided seed is not perfect");

  struct DummyIndexer {
    constexpr DummyIndexer(const std::array<T, kSize>&, uint32_t) {}
  };

  static constexpr std::conditional_t<
      kIsDense, DummyIndexer,
      ConstexprPerfectHashIndexer<T, kSize, kCaseSensitive, kTableSize>>
      kIndexer{kValues, kSeed};

  constexpr DomainField(T value) : index_(GetIndex(value)) {}

  // Enables implicit conversion from types convertible to T
  // (e.g., const char* for string_view domains).
  template <typename U, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<U>, DomainField> &&
                            std::is_convertible_v<U, T>>>
  constexpr DomainField(U&& value)
      : index_(GetIndex(static_cast<T>(std::forward<U>(value)))) {}

  constexpr DomainField(MetricIndex index)
      : index_(GetIndexFromMetricIndex(index)) {}

  constexpr size_t index() const { return index_; }
  constexpr bool valid() const { return index_ != kInvalidMetricIndex; }

  constexpr T value() const {
    if (valid()) {
      if constexpr (kIsDense) {
        return static_cast<T>(static_cast<UnderlyingT>(index_) +
                              static_cast<UnderlyingT>(kMin));
      } else {
        return kIndexer.GetKey(index_);
      }
    }
    return T{};
  }

  constexpr operator T() const { return value(); }

  template <typename Dummy = T, typename U = UnderlyingTypeOrSelfT<Dummy>,
            typename = std::enable_if_t<std::is_enum_v<Dummy>>>
  explicit constexpr operator U() const {
    return static_cast<U>(value());
  }

  constexpr bool operator==(const DomainField& other) const {
    return index_ == other.index_;
  }
  constexpr bool operator!=(const DomainField& other) const {
    return index_ != other.index_;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DomainField& df) {
    if (!df.valid()) {
      return;
    }
    absl::Format(&sink, "%v", df.value());
  }

  template <typename H>
  friend H AbslHashValue(H h, const DomainField& df) {
    return H::combine(std::move(h), df.value());
  }

 private:
  static constexpr size_t GetIndexFromMetricIndex(MetricIndex idx) {
    if (idx.value >= kTableSize) return kInvalidMetricIndex;
    if constexpr (kIsDense) {
      return idx.value;
    } else {
      return kIndexer.occupied(idx.value) ? idx.value : kInvalidMetricIndex;
    }
  }

  static constexpr size_t GetIndex(T value) {
    if constexpr (kIsDense) {
      UnderlyingT val_idx = static_cast<UnderlyingT>(value);
      UnderlyingT min_idx = static_cast<UnderlyingT>(kMin);
      UnderlyingT max_idx = static_cast<UnderlyingT>(kMax);
      if (val_idx >= min_idx && val_idx <= max_idx) {
        return static_cast<size_t>(val_idx - min_idx);
      }
      return kInvalidMetricIndex;
    } else {
      return kIndexer.GetIndex(value);
    }
  }

  size_t index_;
};

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_DOMAIN_FIELD_H_
