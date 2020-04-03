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

#ifndef TENSORSTORE_UTIL_ITERATE_OVER_INDEX_RANGE_H_
#define TENSORSTORE_UTIL_ITERATE_OVER_INDEX_RANGE_H_

#include <cassert>
#include <type_traits>
#include <utility>

#include "absl/container/fixed_array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/void_wrapper.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/default_iteration_result.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_iterate {

/// Returns the index of the dimension over which to loop within the inner loop
/// of `outer_dims` dimensions.
inline constexpr DimensionIndex GetLoopDimension(ContiguousLayoutOrder order,
                                                 DimensionIndex outer_dims,
                                                 DimensionIndex total_dims) {
  return order == ContiguousLayoutOrder::c ? outer_dims
                                           : total_dims - 1 - outer_dims;
}

template <typename Func, typename IndexType, DimensionIndex Rank>
using IterateOverIndexRangeResult =
    std::decay_t<std::invoke_result_t<Func, span<const IndexType, Rank>>>;

template <ContiguousLayoutOrder Order, typename Func, typename IndexType,
          DimensionIndex Rank>
struct IterateOverIndexRangeHelper {
  using IndicesSpan = span<const IndexType, Rank>;
  using ResultType = IterateOverIndexRangeResult<Func, IndexType, Rank>;
  using WrappedResultType = internal::Void::WrappedType<ResultType>;

  static WrappedResultType Loop(Func func, DimensionIndex outer_dims,
                                const IndexType* origin, const IndexType* shape,
                                span<IndexType, Rank> indices) {
    WrappedResultType result =
        DefaultIterationResult<WrappedResultType>::value();
    const DimensionIndex cur_dim =
        GetLoopDimension(Order, outer_dims, indices.size());
    const IndexType start = origin[cur_dim];
    const IndexType stop = shape[cur_dim] + start;
    if (outer_dims + 1 == indices.size()) {
      for (IndexType i = start; i < stop; ++i) {
        indices[cur_dim] = i;
        result = internal::Void::CallAndWrap(func, IndicesSpan(indices));
        if (!result) break;
      }
    } else {
      for (IndexType i = start; i < stop; ++i) {
        indices[cur_dim] = i;
        result = Loop(func, outer_dims + 1, origin, shape, indices);
        if (!result) break;
      }
    }
    return result;
  }
  static ResultType Start(Func func, const IndexType* origin,
                          IndicesSpan shape) {
    if (shape.size() == 0) {
      return func(span<const IndexType, Rank>());
    }
    absl::FixedArray<IndexType, internal::kNumInlinedDims> indices(
        shape.size());
    return internal::Void::Unwrap(
        Loop(func, 0, &origin[0], &shape[0], indices));
  }
};
}  // namespace internal_iterate

/// Iterates over a multi-dimensional hyperrectangle specified by `origin` and
/// `shape` and invokes `func` with a `span<const Index, Rank>` of indices
/// corresponding to each position.
///
/// For example:
///
/// `IterateOverIndexRange<c_order>(span({0, 0}), span({2, 3}), func)`
/// invokes:
///
///     `func({0, 0})`, `func({0, 1})`, `func({0, 2})`,
///     `func({1, 0})`, `func({1, 1})`, `func({1, 2})`.
///
/// `IterateOverIndexRange<fortran_order>(span({0, 0}),
///                                       span({2, 3}),
///                                       func)`
/// invokes:
///
///     `func({0, 0})`, `func({1, 0})`,
///     `func({0, 1})`, `func({1, 1})`,
///     `func({0, 2})`, `func({1, 2})`.
///
/// `IterateOverIndexRange<c_order>(span({0, 1}), span({2, 2}), func)`
/// invokes:
///
///     `func({0, 1})`, `func({0, 2})`,
///     `func({1, 1})`, `func({1, 2})`.
///
/// \tparam Order The order in which to iterate.
/// \param origin The origin from which iteration starts.
/// \param shape The multi-dimensional shape over which this function iterates.
/// \param func The function to invoke for each position.  It must be invocable
///     as `func(std::declval<span<const IndexType, Rank>>())`, and the return
///     type must be `void` or a default-constructible type `ResultType`
///     explicitly convertible to `bool`.  A non-void return convertible to
///     `false` causes iteration to stop.
/// \dchecks `origin.size() == shape.size()`.
/// \returns `void` if `func` returns `void`.  Otherwise, returns the result of
///     the last invocation of `func`, or
///     `DefaultIterationResult<ResultType>::value()` if `shape` contains an
///     extent of `0`.
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename IndexType, DimensionIndex Rank, typename Func>
internal_iterate::IterateOverIndexRangeResult<
    Func, std::remove_const_t<IndexType>, Rank>
IterateOverIndexRange(span<IndexType, Rank> origin, span<IndexType, Rank> shape,
                      Func&& func) {
  assert(origin.size() == shape.size());
  return internal_iterate::IterateOverIndexRangeHelper<
      Order, Func, std::remove_const_t<IndexType>, Rank>::Start(func,
                                                                origin.data(),
                                                                shape);
}

/// Equivalent to:
/// `IterateOverIndexRange<Order>(box.origin(), box.shape(), func)`.
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename BoxType, typename Func>
std::enable_if_t<IsBoxLike<BoxType>::value,
                 internal_iterate::IterateOverIndexRangeResult<
                     Func, Index, BoxType::static_rank>>
IterateOverIndexRange(const BoxType& box, Func&& func,
                      ContiguousLayoutOrder order = ContiguousLayoutOrder::c) {
  return internal_iterate::IterateOverIndexRangeHelper<
      Order, Func, Index, BoxType::static_rank>::Start(func,
                                                       box.origin().data(),
                                                       box.shape());
}

/// Iterates over a multi-dimensional `shape` and invokes `func` with a
/// `span<const Index, Rank>` of indices corresponding to each position.
///
/// Equivalent to:
/// `IterateOverIndexRange(GetConstantVector<IndexType, 0>(shape.size()),
///                        shape, func, order)`.
///
/// For example:
///
/// `IterateOverIndexRange<c_order>(span({2, 3}), func)` invokes:
///
///     `func({0, 0})`, `func({0, 1})`, `func({0, 2})`,
///     `func({1, 0})`, `func({1, 1})`, `func({1, 2})`.
///
/// `IterateOverIndexRange<fortran_order>(span({2, 3}), func)` invokes:
///
///     `func({0, 0})`, `func({1, 0})`,
///     `func({0, 1})`, `func({1, 1})`,
///     `func({0, 2})`, `func({1, 2})`.
///
/// \tparam Order The order in which to iterate.
/// \param shape The multi-dimensional shape over which this function iterates.
/// \param func The function to invoke for each position.  It must be invocable
///     as `func(std::declval<span<const IndexType, Rank>>())`, and the return
///     type must be `void` or a default-constructible type `ResultType`
///     explicitly convertible to `bool`.  A non-void return convertible to
///     `false` causes iteration to stop.
/// \returns `void` if `func` returns `void`.  Otherwise, returns the result of
///     the last invocation of `func`, or
///     `DefaultIterationResult<ResultType>::value()` if `shape` contains an
///     extent of `0`.
template <ContiguousLayoutOrder Order = ContiguousLayoutOrder::c,
          typename IndexType, DimensionIndex Rank, typename Func>
internal_iterate::IterateOverIndexRangeResult<
    Func, std::remove_const_t<IndexType>, Rank>
IterateOverIndexRange(span<IndexType, Rank> shape, Func&& func) {
  using NonConstIndex = std::remove_const_t<IndexType>;
  return internal_iterate::
      IterateOverIndexRangeHelper<Order, Func, NonConstIndex, Rank>::Start(
          func,
          GetConstantVector<NonConstIndex, 0>(GetStaticOrDynamicExtent(shape))
              .data(),
          shape);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ITERATE_OVER_INDEX_RANGE_H_
