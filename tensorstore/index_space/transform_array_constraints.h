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

#ifndef TENSORSTORE_INDEX_SPACE_TRANSFORM_ARRAY_CONSTRAINTS_H_
#define TENSORSTORE_INDEX_SPACE_TRANSFORM_ARRAY_CONSTRAINTS_H_

#include "tensorstore/util/iterate.h"

namespace tensorstore {

/// Specifies whether a new array must be allocated.
///
/// \relates TransformArrayConstraints
enum class MustAllocateConstraint {
  /// A reference to the existing array may be returned, or a new array may be
  /// allocated if necessary.
  may_allocate = 0,
  /// A new array will always be allocated.
  must_allocate = 1
};

/// \relates MustAllocateConstraint
constexpr MustAllocateConstraint may_allocate =
    MustAllocateConstraint::may_allocate;
constexpr MustAllocateConstraint must_allocate =
    MustAllocateConstraint::must_allocate;

/// Specifies constraints on the array returned from `TransformArray`.
///
/// This type is the algebraic product of `IterationConstraints` and
/// `MustAllocateConstraint`.
///
/// A `MustAllocateConstraint` of `must_allocate` indicates that a newly
/// allocated array is always returned, subject to the `IterationConstraints`.
/// Otherwise, the returned array may point to the existing array if possible.
/// (If `may_allocate` is specified but a new array is allocated, the newly
/// allocated array is still subject to the `IterationConstraints`.)
///
/// \ingroup array transformation
class TransformArrayConstraints {
 public:
  /// Constructs from various combinations of constraints.
  constexpr TransformArrayConstraints(
      IterationConstraints iteration_constraint = {},
      MustAllocateConstraint allocate_constraint = may_allocate)
      : value_(iteration_constraint.value() |
               (static_cast<int>(allocate_constraint)
                << IterationConstraints::kNumBits)) {}
  constexpr TransformArrayConstraints(
      LayoutOrderConstraint order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      UnspecifiedLayoutOrder order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      ContiguousLayoutOrder order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(
            IterationConstraints(order_constraint, repeat_constraint),
            allocate_constraint) {}
  constexpr TransformArrayConstraints(
      LayoutOrderConstraint order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      UnspecifiedLayoutOrder order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      ContiguousLayoutOrder order_constraint,
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints(order_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      RepeatedElementsConstraint repeat_constraint,
      MustAllocateConstraint allocate_constraint = may_allocate)
      : TransformArrayConstraints(IterationConstraints(repeat_constraint),
                                  allocate_constraint) {}
  constexpr TransformArrayConstraints(
      MustAllocateConstraint allocate_constraint)
      : TransformArrayConstraints(IterationConstraints{}, allocate_constraint) {
  }

  explicit constexpr TransformArrayConstraints(int value) : value_(value) {}

  /// Returns the iteration constraints.
  ///
  /// \membergroup Accessors
  constexpr IterationConstraints iteration_constraints() const {
    return IterationConstraints(value() &
                                ((1 << IterationConstraints::kNumBits) - 1));
  }

  /// Returns the order constraint.
  ///
  /// \membergroup Accessors
  constexpr LayoutOrderConstraint order_constraint() const {
    return iteration_constraints().order_constraint();
  }

  /// Returns the repeated elements constraint.
  ///
  /// \membergroup Accessors
  constexpr RepeatedElementsConstraint repeated_elements_constraint() const {
    return iteration_constraints().repeated_elements_constraint();
  }

  /// Returns the allocation constraint.
  ///
  /// \membergroup Accessors
  constexpr MustAllocateConstraint allocate_constraint() const {
    return static_cast<MustAllocateConstraint>(value_ >>
                                               IterationConstraints::kNumBits);
  }

  constexpr int value() const { return value_; }
  constexpr static int kNumBits = IterationConstraints::kNumBits + 1;

  /// Compares two constraints.
  friend constexpr bool operator==(TransformArrayConstraints a,
                                   TransformArrayConstraints b) {
    return a.value() == b.value();
  }
  friend constexpr bool operator!=(TransformArrayConstraints a,
                                   TransformArrayConstraints b) {
    return a.value() != b.value();
  }

 private:
  int value_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_TRANSFORM_ARRAY_CONSTRAINTS_H_
