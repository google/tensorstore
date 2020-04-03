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

#ifndef TENSORSTORE_INDEX_H_
#define TENSORSTORE_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace tensorstore {

/// Type for representing a coordinate in, or a extent of, a multi-dimensional
/// array.
using Index = std::int64_t;

/// Type for representing a dimension index or rank.
using DimensionIndex = std::ptrdiff_t;

/// A valid index is in the closed interval:
/// `[kMinFiniteIndex, kMaxFiniteIndex]`.

/// The special value of `-kInfIndex` when used as a lower bound indicates an
/// index range that is unbounded below, and the special value of `+kInfIndex`
/// when used as an upper bound indicates an index range that is unbounded
/// above.
constexpr Index kMinFiniteIndex = -0x3ffffffffffffffe;  // -(2^62-2)
constexpr Index kInfIndex = 0x3fffffffffffffff;         // 2^62-1
constexpr Index kMaxFiniteIndex = 0x3ffffffffffffffe;   // 2^62-2
constexpr Index kInfSize = 0x7fffffffffffffff;          // 2^63-1
constexpr Index kMaxFiniteSize = 0x7ffffffffffffffd;    // 2^63-3
static_assert(-kInfIndex + kInfSize - 1 == kInfIndex, "");
static_assert(kMinFiniteIndex + kMaxFiniteSize - 1 == kMaxFiniteIndex, "");

/// Special index value that indicates an implicit bound.
constexpr Index kImplicit = -0x8000000000000000;  // -2^63

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_H_
