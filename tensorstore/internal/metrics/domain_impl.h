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

#ifndef TENSORSTORE_INTERNAL_METRICS_DOMAIN_IMPL_H_
#define TENSORSTORE_INTERNAL_METRICS_DOMAIN_IMPL_H_

// Implementation details for internal/metrics domain fields.
// IWYU pragma: private

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "tensorstore/internal/meta/requires.h"

namespace tensorstore {
namespace internal_metrics {

inline constexpr size_t kInvalidMetricIndex = static_cast<size_t>(-1);

template <typename T, bool IsEnum = std::is_enum_v<T>>
struct UnderlyingTypeOrSelf {
  using type = T;
};

template <typename T>
struct UnderlyingTypeOrSelf<T, true> {
  using type = std::underlying_type_t<T>;
};

// Returns the underlying type of an enum or T itself if T is not an enum.
template <typename T>
using UnderlyingTypeOrSelfT = typename UnderlyingTypeOrSelf<T>::type;

// HasSeed trait detector detects T::kSeed.
template <typename T>
inline constexpr bool HasSeedV = internal_meta::Requires<T>(
    [](auto&& x) -> decltype(std::decay_t<decltype(x)>::kSeed) {});

template <typename T>
inline constexpr bool HasTableSizeV = internal_meta::Requires<T>(
    [](auto&& x) -> decltype(std::decay_t<decltype(x)>::kTableSize) {});

template <typename T>
inline constexpr bool HasCaseSensitiveV = internal_meta::Requires<T>(
    [](auto&& x) -> decltype(std::decay_t<decltype(x)>::kCaseSensitive) {});

template <typename Spec, bool Default>
inline constexpr bool GetCaseSensitiveV = [] {
  if constexpr (HasCaseSensitiveV<Spec>) {
    return Spec::kCaseSensitive;
  } else {
    return Default;
  }
}();

constexpr char ToLower(char c) {
  return (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c;
}

template <bool CaseSensitive>
constexpr bool Equals(std::string_view a, std::string_view b) {
  if constexpr (CaseSensitive) {
    return a == b;
  } else {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (ToLower(a[i]) != ToLower(b[i])) return false;
    }
    return true;
  }
}

template <typename T, size_t N, bool CaseSensitive = false>
constexpr bool HasDuplicates(const std::array<T, N>& keys) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      if constexpr (std::is_same_v<T, std::string_view>) {
        if (Equals<CaseSensitive>(keys[i], keys[j])) return true;
      } else {
        if (keys[i] == keys[j]) return true;
      }
    }
  }
  return false;
}

// Modified FNV-1a hash with parameterized seed and MurmurHash3 finalizer
// for constexpr use, parameterized case-sensitivity.
template <bool CaseSensitive>
constexpr uint32_t FnvHash(std::string_view sv, uint32_t seed) {
  uint32_t hash = seed;
  for (char c : sv) {
    if constexpr (CaseSensitive) {
      hash ^= static_cast<uint8_t>(c);
    } else {
      hash ^= static_cast<uint8_t>(ToLower(c));
    }
    hash *= 16777619u;  // FNV-1a prime
  }
  hash ^= hash >> 13;  // MurmurHash3 finalizer mixing
  hash *= 0x85ebca6bu;
  hash ^= hash >> 16;
  return hash;
}

// Modified FNV hash with parameterized seed and MurmurHash3 finalizer
// for integers and enums. CaseSensitive is accepted for call-site
// uniformity but has no effect.
template <bool CaseSensitive = false, typename T>
constexpr uint32_t FnvHash(T val, uint32_t seed) {
  static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
  using U = UnderlyingTypeOrSelfT<T>;
  U u_val = static_cast<U>(val);
  uint32_t hash = seed;
  for (size_t i = 0; i < sizeof(U); ++i) {
    hash ^= static_cast<uint8_t>(u_val >> (i * 8));
    hash *= 16777619u;  // FNV-1a prime
  }
  hash ^= hash >> 13;  // MurmurHash3 finalizer mixing
  hash *= 0x85ebca6bu;
  hash ^= hash >> 16;
  return hash;
}

// Pointer-based IsPerfect, reusable at runtime.
// Precondition: used_buf[0..table_size) is zero-initialized.
// used_buf must have at least table_size elements.
template <typename T, bool CaseSensitive = false, typename BoolType>
constexpr bool IsPerfect(const T* keys, size_t n, uint32_t seed,
                         BoolType* used_buf, size_t table_size) {
  for (size_t i = 0; i < n; ++i) {
    const auto& key = keys[i];
    uint32_t h = FnvHash<CaseSensitive>(key, seed) % table_size;
    if (used_buf[h]) return false;
    used_buf[h] = 1;
  }
  return true;
}

// Replace with std::optional<uint32_t> in C++20 (P2231R1).
struct ConstexprSeed {
  bool has_val;
  uint32_t val;
  constexpr bool has_value() const { return has_val; }
  constexpr uint32_t value() const { return val; }
  constexpr uint32_t operator*() const { return val; }

  constexpr operator std::optional<uint32_t>() const {
    return has_val ? std::optional<uint32_t>(val) : std::nullopt;
  }
};

// std::array wrapper for IsPerfect.
template <typename T, size_t N, bool CaseSensitive, size_t TableSize>
constexpr bool IsPerfect(const std::array<T, N>& keys, uint32_t seed) {
  bool used[TableSize] = {};
  return IsPerfect<T, CaseSensitive>(keys.data(), N, seed, used, TableSize);
}

// Constexpr perfect hash indexer using non-minimal table.
template <typename T, size_t N, bool CaseSensitive, size_t TableSize>
class ConstexprPerfectHashIndexer {
 public:
  static_assert(std::is_integral_v<T> || std::is_enum_v<T> ||
                    std::is_same_v<T, std::string_view>,
                "Unsupported type for ConstexprPerfectHashIndexer");

  constexpr ConstexprPerfectHashIndexer(const std::array<T, N>& keys,
                                        uint32_t seed)
      : keys_(), occupied_(), seed_(seed) {
    for (const auto& key : keys) {
      uint32_t h = FnvHash<CaseSensitive>(key, seed) % TableSize;
      keys_[h] = key;
      occupied_[h] = true;
    }
  }

  constexpr size_t GetIndex(T key) const {
    uint32_t h = FnvHash<CaseSensitive>(key, seed_) % TableSize;
    if constexpr (std::is_same_v<T, std::string_view>) {
      if (occupied_[h] && Equals<CaseSensitive>(keys_[h], key)) return h;
    } else {
      if (occupied_[h] && keys_[h] == key) return h;
    }
    return kInvalidMetricIndex;
  }

  constexpr T GetKey(size_t index) const { return keys_[index]; }
  constexpr bool occupied(size_t index) const { return occupied_[index]; }

 private:
  std::array<T, TableSize> keys_;
  std::array<bool, TableSize> occupied_;
  uint32_t seed_;
};

template <typename T, size_t N>
constexpr T FindMin(const std::array<T, N>& values) {
  static_assert(N > 0, "Array cannot be empty");
  T min_val = values[0];
  for (size_t i = 1; i < N; ++i) {
    if (values[i] < min_val) min_val = values[i];
  }
  return min_val;
}

template <typename T, size_t N>
constexpr T FindMax(const std::array<T, N>& values) {
  static_assert(N > 0, "Array cannot be empty");
  T max_val = values[0];
  for (size_t i = 1; i < N; ++i) {
    if (values[i] > max_val) max_val = values[i];
  }
  return max_val;
}

template <typename T, size_t N>
constexpr bool IsDenseSequential(const std::array<T, N>& values) {
  if constexpr (std::is_enum_v<T> || std::is_integral_v<T>) {
    T min_val = FindMin(values);
    T max_val = FindMax(values);
    using U = UnderlyingTypeOrSelfT<T>;
    U min_idx = static_cast<U>(min_val);
    U max_idx = static_cast<U>(max_val);
    if (max_idx < min_idx) return false;
    using UnsignedU = std::make_unsigned_t<
        std::conditional_t<std::is_same_v<U, bool>, int, U>>;
    UnsignedU diff =
        static_cast<UnsignedU>(max_idx) - static_cast<UnsignedU>(min_idx);
    if (diff >= N) return false;
    if (static_cast<size_t>(diff) + 1 != N) return false;
    return !HasDuplicates<T, N>(values);
  }
  return false;
}

template <typename Cell>
bool IsDefaultCell(const Cell& cell) {
  if constexpr (internal_meta::Requires<const Cell&>(
                    [](auto&& c) -> decltype(c.GetCount()) {})) {
    return cell.GetCount() == 0;
  } else {
    return cell.Get() == typename Cell::value_type{};
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_DOMAIN_IMPL_H_
