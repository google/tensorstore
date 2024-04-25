// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_ARRAY_STORAGE_STATISTICS_H_
#define TENSORSTORE_ARRAY_STORAGE_STATISTICS_H_

#include <iosfwd>
#include <optional>
#include <type_traits>

#include "absl/time/time.h"
#include "tensorstore/batch.h"

namespace tensorstore {

struct ArrayStorageStatistics {
  enum Mask {
    /// Query if no data is stored.
    query_not_stored = 1,

    /// Query if data is stored for all elements in the requested domain.
    query_fully_stored = 2,
  };

  friend constexpr Mask operator~(Mask a) {
    return static_cast<Mask>(~static_cast<std::underlying_type_t<Mask>>(a));
  }

  friend constexpr Mask operator|(Mask a, Mask b) {
    using U = std::underlying_type_t<Mask>;
    return static_cast<Mask>(static_cast<U>(a) | static_cast<U>(b));
  }

  friend constexpr Mask& operator|=(Mask& a, Mask b) {
    using U = std::underlying_type_t<Mask>;
    return a = static_cast<Mask>(static_cast<U>(a) | static_cast<U>(b));
  }

  friend constexpr Mask operator&(Mask a, Mask b) {
    using U = std::underlying_type_t<Mask>;
    return static_cast<Mask>(static_cast<U>(a) & static_cast<U>(b));
  }

  friend constexpr Mask operator&=(Mask& a, Mask b) {
    using U = std::underlying_type_t<Mask>;
    return a = static_cast<Mask>(static_cast<U>(a) & static_cast<U>(b));
  }

  /// Indicates which fields are valid.
  Mask mask = {};

  /// Set to `true` if no data is stored within the requested domain.
  ///
  /// Only valid if `mask` includes `query_not_stored`.
  ///
  /// - `true` indicates no data is stored for the requested domain.
  ///
  /// - `false` indicates some data is stored for the requested domain.
  bool not_stored = false;

  /// Indicates whether the requested domain is fully stored.
  ///
  /// Only valid if `mask` includes `query_fully_stored`.
  ///
  /// - `true` indicates the requested domain is fully stored.
  ///
  /// - `false` indicates at least some portion of the requested domain is not
  ///   stored.
  bool fully_stored = false;

  /// Comparison operators.
  friend bool operator==(const ArrayStorageStatistics& a,
                         const ArrayStorageStatistics& b);
  friend bool operator!=(const ArrayStorageStatistics& a,
                         const ArrayStorageStatistics& b) {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ArrayStorageStatistics& a);

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.mask, x.not_stored, x.fully_stored);
  };
};

struct GetArrayStorageStatisticsOptions {
  ArrayStorageStatistics::Mask mask = {};

  Batch batch{no_batch};

  void Set(ArrayStorageStatistics::Mask value) { mask |= value; }

  void Set(Batch value) { this->batch = std::move(value); }

  template <typename T>
  constexpr static inline bool IsOption = false;
};

template <>
constexpr inline bool
    GetArrayStorageStatisticsOptions::IsOption<ArrayStorageStatistics::Mask> =
        true;

template <>
constexpr inline bool GetArrayStorageStatisticsOptions::IsOption<Batch> = true;

template <>
constexpr inline bool GetArrayStorageStatisticsOptions::IsOption<Batch::View> =
    true;

}  // namespace tensorstore

#endif  // TENSORSTORE_ARRAY_STORAGE_STATISTICS_H_
