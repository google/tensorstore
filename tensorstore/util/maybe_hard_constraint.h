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

#ifndef TENSORSTORE_UTIL_MAYBE_HARD_CONSTRAINT_H_
#define TENSORSTORE_UTIL_MAYBE_HARD_CONSTRAINT_H_

#include "tensorstore/index.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Combines a span (with length assumed to be a valid rank) with a bit vector
/// indicating which values are hard constraints as opposed to soft constraints.
///
/// This type serves as a base class for option types like
/// `ChunkLayout::GridOrigin`.
///
/// \ingroup utilities
template <typename T>
class MaybeHardConstraintSpan : public span<const T> {
 public:
  /// Constructs a zero-length span with no hard constraints.
  ///
  /// This is equivalent to specifying no constraints.
  ///
  /// \id default
  explicit MaybeHardConstraintSpan() = default;

  /// Constructs from a vector and hard constraint dimension set.
  ///
  /// \id span, hard_constraint
  explicit MaybeHardConstraintSpan(
      span<const T> s, DimensionSet hard_constraint = DimensionSet(true))
      : span<const T>(s), hard_constraint(hard_constraint) {}
  template <size_t N>
  explicit MaybeHardConstraintSpan(
      const T (&s)[N], DimensionSet hard_constraint = DimensionSet(true))
      : span<const T>(s), hard_constraint(hard_constraint) {}

  /// Checks if the `span` has non-zero length.
  constexpr bool valid() const { return !this->empty(); }

  /// Compares two constraints for equality.
  friend bool operator==(const MaybeHardConstraintSpan& a,
                         const MaybeHardConstraintSpan& b) {
    return internal::RangesEqual(a, b) &&
           a.hard_constraint == b.hard_constraint;
  }
  friend bool operator!=(const MaybeHardConstraintSpan& a,
                         const MaybeHardConstraintSpan& b) {
    return !(a == b);
  }

  /// Set of dimensions for which the corresponding constraint value should be
  /// treated as a hard constraint.
  DimensionSet hard_constraint;
};

/// Combines an `Index` (which defaults to `kImplicit`) with a `bool` indicating
/// whether the value is a hard constraint as opposed to a default value.
///
/// This type serves as a base class for option types like
/// `ChunkLayout::ChunkElements`.
///
/// \ingroup utilities
class MaybeHardConstraintIndex {
 public:
  using value_type = Index;

  /// Constructs from a value.
  explicit MaybeHardConstraintIndex(Index value = kImplicit,
                                    bool hard_constraint = true)
      : value(value), hard_constraint(hard_constraint) {}

  /// Implicitly converts to the `Index` value.
  constexpr operator Index() const { return value; }

  /// Checks if the constraint is specified, i.e. `value != kImplicit`.
  bool valid() const { return value != kImplicit; }

  /// Compares two constraints for equality.
  friend bool operator==(const MaybeHardConstraintIndex& a,
                         const MaybeHardConstraintIndex& b) {
    return a.value == b.value && a.hard_constraint == b.hard_constraint;
  }
  friend bool operator!=(const MaybeHardConstraintIndex& a,
                         const MaybeHardConstraintIndex& b) {
    return !(a == b);
  }

  /// `Index` value.
  Index value;

  /// Indicates whether `value` should be treated as a hard constraint.
  bool hard_constraint;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_MAYBE_HARD_CONSTRAINT_H_
