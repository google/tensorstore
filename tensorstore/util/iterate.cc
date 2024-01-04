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

#include "tensorstore/util/iterate.h"

#include <stddef.h>

#include <algorithm>
#include <array>
#include <ostream>
#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/utility/utility.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/internal/iterate.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

namespace internal_iterate {

template <size_t Arity>
static bool AreStridesContiguous(
    const InnerShapeAndStrides<Arity, 2>& inner_layout,
    const std::array<ptrdiff_t, Arity>& element_sizes) {
  if (inner_layout.shape[1] > 1) {
    for (size_t i = 0; i < Arity; ++i) {
      if (inner_layout.strides[i][1] != element_sizes[i]) return false;
    }
  }
  return true;
}

absl::InlinedVector<DimensionIndex, internal::kNumInlinedDims>
ComputeStridedLayoutDimensionIterationOrder(IterationConstraints constraints,
                                            span<const Index> shape,
                                            span<const Index* const> strides) {
  // TODO(jbms): Consider changing this function back to be templated on
  // `strides.size()` and/or `constraints`.
  const DimensionIndex rank = shape.size();

  absl::InlinedVector<DimensionIndex, internal::kNumInlinedDims>
      dimension_order(rank);
  {
    DimensionIndex num_dims_preserved = 0;
    for (DimensionIndex dim_i = 0; dim_i < rank; ++dim_i) {
      const Index size = shape[dim_i];
      // Skip dimensions of size 1, as they can safely be ignored.
      if (size == 1) continue;

      // If we can skip repeated elements, skip dimensions for which all stride
      // values are 0.
      if (size != 0 && constraints.repeated_elements_constraint() ==
                           skip_repeated_elements) {
        for (std::ptrdiff_t i = 0; i < strides.size(); ++i) {
          if (strides[i][dim_i] != 0) goto cannot_skip_dimension;
        }
        continue;
      }

    cannot_skip_dimension:
      dimension_order[num_dims_preserved++] = dim_i;
    }
    dimension_order.resize(num_dims_preserved);
  }

  if (constraints.order_constraint()) {
    if (constraints.order_constraint() == ContiguousLayoutOrder::fortran) {
      std::reverse(dimension_order.begin(), dimension_order.end());
    }
  } else {
    std::sort(dimension_order.begin(), dimension_order.end(),
              [&](DimensionIndex a, DimensionIndex b) {
                for (ptrdiff_t j = 0; j < strides.size(); ++j) {
                  const Index stride_a = strides[j][a];
                  const Index stride_b = strides[j][b];
                  if (stride_a > stride_b) return true;
                  if (stride_a < stride_b) return false;
                }
                return false;
              });
  }
  return dimension_order;
}

}  // namespace internal_iterate

namespace internal {

template <size_t Arity>
static SpecializedElementwiseFunctionPointer<Arity, void*>
PickElementwiseFunction(
    const internal_iterate::InnerShapeAndStrides<Arity, 2>& inner_layout,
    const ElementwiseFunction<Arity, void*>& function,
    std::array<std::ptrdiff_t, Arity> element_sizes) {
  return function[internal_iterate::AreStridesContiguous(inner_layout,
                                                         element_sizes)
                      ? IterationBufferKind::kContiguous
                      : IterationBufferKind::kStrided];
}

template <size_t Arity>
StridedLayoutFunctionApplyer<Arity>::StridedLayoutFunctionApplyer(
    span<const Index> shape, std::array<const Index*, Arity> strides,
    IterationConstraints constraints, ElementwiseClosure<Arity, void*> closure,
    std::array<std::ptrdiff_t, Arity> element_sizes)
    : iteration_layout_(internal_iterate::SimplifyStridedIterationLayout(
          constraints, shape, strides)),
      inner_layout_(
          internal_iterate::ExtractInnerShapeAndStrides<2>(&iteration_layout_)),
      context_(closure.context),
      callback_(PickElementwiseFunction(inner_layout_, *closure.function,
                                        element_sizes)) {}

template <size_t Arity>
StridedLayoutFunctionApplyer<Arity>::StridedLayoutFunctionApplyer(
    const Index* shape, span<const DimensionIndex> dimension_order,
    std::array<const Index*, Arity> strides,
    ElementwiseClosure<Arity, void*> closure,
    std::array<std::ptrdiff_t, Arity> element_sizes)
    : iteration_layout_(
          internal_iterate::PermuteAndSimplifyStridedIterationLayout(
              shape, dimension_order, strides)),
      inner_layout_(
          internal_iterate::ExtractInnerShapeAndStrides<2>(&iteration_layout_)),
      context_(closure.context),
      callback_(PickElementwiseFunction(inner_layout_, *closure.function,
                                        element_sizes)) {}

template <size_t Arity>
struct StridedLayoutFunctionApplyer<Arity>::WrappedFunction {
  template <typename... Pointer>
  bool operator()(Pointer... pointer) const {
    return CallHelper(std::index_sequence_for<Pointer...>(), pointer...);
  }

  template <size_t... Is>
  static bool OuterCallHelper(
      const StridedLayoutFunctionApplyer& data, std::index_sequence<Is...>,
      std::array<ByteStridedPointer<void>, Arity> pointers, void* arg) {
    return internal_iterate::IterateHelper<
        WrappedFunction,
        std::enable_if_t<true || Is, ByteStridedPointer<void>>...>::
        Start(WrappedFunction{data, arg}, data.iteration_layout_,
              pointers[Is]...);
  }

  template <size_t... Is, typename... Pointer>
  bool CallHelper(std::index_sequence<Is...>, Pointer... pointer) const {
    return data_.callback_(
        data_.context_, data_.inner_layout_.shape,
        IterationBufferPointer{pointer, data_.inner_layout_.strides[Is][0],
                               data_.inner_layout_.strides[Is][1]}...,
        arg_);
  }

  const StridedLayoutFunctionApplyer& data_;
  void* arg_;
};

template <size_t Arity>
bool StridedLayoutFunctionApplyer<Arity>::operator()(
    std::array<ByteStridedPointer<void>, Arity> pointers, void* arg) const {
  return WrappedFunction::OuterCallHelper(
      *this, std::make_index_sequence<Arity>(), pointers, arg);
}

template <size_t Arity>
bool IterateOverStridedLayouts(
    ElementwiseClosure<Arity, void*> closure, void* arg,
    span<const Index> shape,
    std::array<ByteStridedPointer<void>, Arity> pointers,
    std::array<const Index*, Arity> strides, IterationConstraints constraints,
    std::array<std::ptrdiff_t, Arity> element_sizes) {
  return StridedLayoutFunctionApplyer<Arity>(
      shape, strides, constraints, closure, element_sizes)(pointers, arg);
}

#define TENSORSTORE_DO_INSTANTIATE_ITERATE(Arity)           \
  template class StridedLayoutFunctionApplyer<Arity>;       \
  template bool IterateOverStridedLayouts<Arity>(           \
      ElementwiseClosure<Arity, void*> closure, void* arg,  \
      span<const Index> shape,                              \
      std::array<ByteStridedPointer<void>, Arity> pointers, \
      std::array<const Index*, Arity> strides,              \
      IterationConstraints constraints,                     \
      std::array<std::ptrdiff_t, Arity> element_sizes);

/**/

TENSORSTORE_INTERNAL_FOR_EACH_ARITY(TENSORSTORE_DO_INSTANTIATE_ITERATE)

#undef TENSORSTORE_DO_INSTANTIATE_ITERATE

}  // namespace internal

}  // namespace tensorstore
