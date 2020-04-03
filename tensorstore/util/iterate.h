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

#ifndef TENSORSTORE_UTIL_ITERATE_H_
#define TENSORSTORE_UTIL_ITERATE_H_

#include <array>
#include <iosfwd>

#include "absl/base/macros.h"
#include "absl/types/optional.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Maximum arity supported by iteration functions.
constexpr std::size_t kMaxSupportedIterationArity = 5;

/// Specifies whether repeated elements may be skipped.
/// \see IterationConstraints
enum class RepeatedElementsConstraint {
  include_repeated_elements = 0,
  skip_repeated_elements = 1,
};
constexpr RepeatedElementsConstraint include_repeated_elements =
    RepeatedElementsConstraint::include_repeated_elements;
constexpr RepeatedElementsConstraint skip_repeated_elements =
    RepeatedElementsConstraint::skip_repeated_elements;

enum class UnspecifiedLayoutOrder { unspecified_order };
constexpr UnspecifiedLayoutOrder unspecified_order =
    UnspecifiedLayoutOrder::unspecified_order;

/// Specifies an order constraint on a strided layout/iteration order,
/// equivalent to an algebraic sum of `ContiguousLayoutOrder` and
/// `unspecified_order`.
///
/// There are three possible values: `c_order` (must be in C order),
/// `fortran_order` (must be in Fortran order), and `unspecified_order` (no
/// constraint on order).
class LayoutOrderConstraint {
 public:
  constexpr LayoutOrderConstraint(UnspecifiedLayoutOrder = unspecified_order)
      : value_(0) {}
  constexpr LayoutOrderConstraint(ContiguousLayoutOrder order)
      : value_(static_cast<int>(order) | 2) {}

  explicit constexpr LayoutOrderConstraint(int value) : value_(value) {}

  /// Returns `true` if the order is constrained.
  constexpr explicit operator bool() const { return value_ & 2; }

  /// Returns the order.
  constexpr ContiguousLayoutOrder order() const {
    return ABSL_ASSERT(static_cast<bool>(*this)),
           static_cast<ContiguousLayoutOrder>(value_ & 1);
  }
  constexpr int value() const { return value_; }
  constexpr static int kNumBits = 2;

  friend constexpr bool operator==(LayoutOrderConstraint a,
                                   LayoutOrderConstraint b) {
    return a.value() == b.value();
  }

  friend constexpr bool operator!=(LayoutOrderConstraint a,
                                   LayoutOrderConstraint b) {
    return a.value() != b.value();
  }

 private:
  int value_;
};

/// Specifies constraints on how multi-dimensional iteration can be performed.
///
/// Logically this type is the algebraic product of `LayoutOrderConstraint` and
/// `RepeatedElementsContraint`.
///
/// A `RepeatedElementsConstraint` of `skip_repeated_elements` indicates that
/// repeated elements (due to a stride of 0) may be processed only once.
/// Otherwise, repeated elements are not skipped.
class IterationConstraints {
 public:
  constexpr IterationConstraints(
      LayoutOrderConstraint order_constraint = unspecified_order,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements)
      : value_(order_constraint.value() |
               (static_cast<int>(repeat_constraint)
                << LayoutOrderConstraint::kNumBits)) {}
  constexpr IterationConstraints(
      UnspecifiedLayoutOrder order_constraint,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements)
      : IterationConstraints(LayoutOrderConstraint(order_constraint),
                             repeat_constraint) {}
  constexpr IterationConstraints(
      ContiguousLayoutOrder order,
      RepeatedElementsConstraint repeat_constraint = include_repeated_elements)
      : IterationConstraints(LayoutOrderConstraint(order), repeat_constraint) {}
  constexpr IterationConstraints(RepeatedElementsConstraint repeat_constraint)
      : IterationConstraints(unspecified_order, repeat_constraint) {}
  explicit constexpr IterationConstraints(int value) : value_(value) {}
  constexpr LayoutOrderConstraint order_constraint() const {
    return LayoutOrderConstraint(value_ & 3);
  }
  constexpr RepeatedElementsConstraint repeated_elements_constraint() const {
    return static_cast<RepeatedElementsConstraint>(
        (value_ >> LayoutOrderConstraint::kNumBits) & 1);
  }
  constexpr int value() const { return value_; }
  constexpr static int kNumBits = LayoutOrderConstraint::kNumBits + 1;

  friend constexpr bool operator==(IterationConstraints a,
                                   IterationConstraints b) {
    return a.value() == b.value();
  }

  friend constexpr bool operator!=(IterationConstraints a,
                                   IterationConstraints b) {
    return a.value() != b.value();
  }

 private:
  int value_;
};

struct ArrayIterateResult {
  /// If `true`, all elements were successfully processed.
  bool success;

  /// The total number of elements successfully processed.
  Index count;

  explicit operator bool() const { return success; }

  friend constexpr bool operator==(ArrayIterateResult a, ArrayIterateResult b) {
    return a.success == b.success && a.count == b.count;
  }
  friend constexpr bool operator!=(ArrayIterateResult a, ArrayIterateResult b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, ArrayIterateResult a);
};

namespace internal {

/// Iterates over strided arrays and invokes a type-erased element-wise function
/// for each position.
///
/// For each `position` within the multi-dimensional `shape`, invokes a
/// type-erased element-wise function with a pointer to the element at
/// `position` in each of the arrays.
///
/// If the element-wise function indicates an error (by returning a count less
/// than the count with which it was called) iteration stops.
///
/// \tparam Arity The arity of the element-wise function, equal to the number of
///     arrays over which to iterate simultaneously.
/// \param closure A representation of the type-erased element-wise function.
/// \param status Status pointer to pass to the elementwise function.
/// \param shape The shape of the multi-dimensional array.
/// \param pointers The base pointers for each array.
/// \param strides The array of strides for each pointer in `pointers`.
/// \param constraints The constraints on the iteration order.  See the
///     documentation of IterationConstraints.
/// \param element_sizes The element sizes (in bytes) corresponding to the
///     element types of the arrays.
template <std::size_t Arity>
ArrayIterateResult IterateOverStridedLayouts(
    ElementwiseClosure<Arity, Status*> closure, Status* status,
    span<const Index> shape,
    std::array<ByteStridedPointer<void>, Arity> pointers,
    std::array<const Index*, Arity> strides, IterationConstraints constraints,
    std::array<std::ptrdiff_t, Arity> element_sizes);

/// Number of dimensions that iteration functions support without heap
/// allocation.
constexpr DimensionIndex kNumInlinedDims = 10;

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_UTIL_ITERATE_H_
