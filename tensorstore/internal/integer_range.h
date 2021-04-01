// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_INTEGER_RANGE_H_
#define TENSORSTORE_INTERNAL_INTEGER_RANGE_H_

#include <type_traits>

namespace tensorstore {
namespace internal {

/// Range-based for-compatible range type for an integer range.
template <typename T>
struct IntegerRange {
  using BaseType =
      std::conditional_t<std::is_enum_v<T>, std::underlying_type_t<T>, T>;
  static_assert(std::is_integral_v<BaseType>);
  using value_type = T;
  using reference = T;

  struct iterator {
    constexpr T operator*() const { return static_cast<T>(value); }
    friend constexpr bool operator==(iterator a, iterator b) {
      return a.value == b.value;
    }
    friend constexpr bool operator!=(iterator a, iterator b) {
      return !(a == b);
    }
    constexpr iterator& operator++() {
      ++value;
      return *this;
    }
    BaseType value;
  };

  constexpr explicit IntegerRange(iterator begin, iterator end)
      : begin_(begin), end_(end) {}

  constexpr static IntegerRange Inclusive(T first, T last) {
    return IntegerRange(iterator{BaseType(first)}, ++iterator{BaseType(last)});
  }

  constexpr iterator begin() const { return begin_; }
  constexpr iterator end() const { return end_; }

  constexpr size_t size() const { return static_cast<size_t>(end_ - begin_); }
  constexpr bool empty() const { return size() == 0; }

 private:
  iterator begin_;
  iterator end_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INTEGER_RANGE_H_
