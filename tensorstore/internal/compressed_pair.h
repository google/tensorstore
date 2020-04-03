// Copyright 2020 The TensorStore Authors
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

/// Compressed pair that does not store empty types as an optimization.

#ifndef TENSORSTORE_INTERNAL_COMPRESSED_PAIR_H_
#define TENSORSTORE_INTERNAL_COMPRESSED_PAIR_H_

#include <type_traits>
#include <utility>
#include "absl/meta/type_traits.h"

namespace tensorstore {
namespace internal {

/// Stores `Second` only.
template <typename First, typename Second>
class CompressedFirstEmptyPair {
 public:
  CompressedFirstEmptyPair() = default;
  template <typename T, typename U>
  explicit CompressedFirstEmptyPair(T&&, U&& u) : second_(std::forward<U>(u)) {
    static_assert(std::is_constructible<First, T&&>(),
                  "First must be constructible from T&&.");
  }
  static constexpr First first() { return {}; }
  Second& second() & { return second_; }
  Second&& second() && { return second_; }
  const Second& second() const& { return second_; }

 private:
  Second second_;
};

/// Stores `First` only.
template <typename First, typename Second>
class CompressedSecondEmptyPair {
 public:
  CompressedSecondEmptyPair() = default;
  template <typename T, typename U>
  explicit CompressedSecondEmptyPair(T&& t, U&&) : first_(std::forward<T>(t)) {
    static_assert(std::is_constructible<Second, U&&>(),
                  "Second must be constructible from U&&.");
  }
  const First& first() const& { return first_; }
  First& first() & { return first_; }
  First&& first() && { return first_; }
  static constexpr Second second() { return {}; }

 private:
  First first_;
};

/// Stores `First` and `Second`.
template <typename First, typename Second>
class CompressedFirstSecondPair {
 public:
  CompressedFirstSecondPair() = default;
  template <typename T, typename U>
  explicit CompressedFirstSecondPair(T&& t, U&& u)
      : first_(std::forward<T>(t)), second_(std::forward<U>(u)) {}
  const First& first() const& { return first_; }
  First& first() & { return first_; }
  First&& first() && { return first_; }
  Second& second() & { return second_; }
  Second&& second() && { return second_; }
  const Second& second() const& { return second_; }

 private:
  First first_;
  Second second_;
};

/// Template alias that evaluates to `CompressedFirstEmptyPair<First, Second>`
/// if `First` is empty, else `CompressedSecondEmptyPair<First, Second>` if
/// `Second` is empty, else `std::pair<First, Second>`.
template <typename First, typename Second>
using CompressedPair = absl::conditional_t<
    std::is_empty<First>::value, CompressedFirstEmptyPair<First, Second>,
    absl::conditional_t<std::is_empty<Second>::value,
                        CompressedSecondEmptyPair<First, Second>,
                        CompressedFirstSecondPair<First, Second>>>;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSED_PAIR_H_
