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

/// Functions used by the implementation of iterate.cc, exposed for unit
/// testing.  This header should not be included except by iterate.cc and
/// iterate_test.cc.

#ifndef TENSORSTORE_INTERNAL_ITERATE_IMPL_H_
#define TENSORSTORE_INTERNAL_ITERATE_IMPL_H_

#include <array>
#include <cstddef>
#include <type_traits>

#include "tensorstore/util/internal/iterate.h"
#include "tensorstore/util/iterate.h"

namespace tensorstore {

namespace internal_iterate {

template <std::size_t N>
bool operator==(const DimensionSizeAndStrides<N>& a,
                const DimensionSizeAndStrides<N>& b) {
  return a.size == b.size && a.strides == b.strides;
}

template <std::size_t N>
bool operator!=(const DimensionSizeAndStrides<N>& a,
                const DimensionSizeAndStrides<N>& b) {
  return !(a == b);
}

template <std::size_t N, DimensionIndex InnerRank>
bool operator==(const InnerShapeAndStrides<N, InnerRank>& a,
                const InnerShapeAndStrides<N, InnerRank>& b) {
  return a.shape == b.shape && a.strides == b.strides;
}

template <std::size_t N, DimensionIndex InnerRank>
bool operator!=(const InnerShapeAndStrides<N, InnerRank>& a,
                const InnerShapeAndStrides<N, InnerRank>& b) {
  return !(a == b);
}

/// \brief Permute and simplify a strided iteration layout by combining adjacent
/// dimensions.
///
/// Logically, permutes the `shape` array and the arrays of strides according to
/// `dimension_order`, producing: `pshape[i] = shape[dimension_order[i]]` and
/// `pstrides[j][i] = strides[j][dimension_order[i]]`.  Then, combines
/// consecutive dimensions in `pshape` and `pstrides` according to the following
/// criteria:
///
/// Dimensions `i...i+m` are combined into a single dimension if, and only if,
/// `pstrides[j][k+1] * pshape[k+1] == pstrides[j][k]` for all `i <= k < m` and
/// `0 <= j < N`.  The resulting combined dimension has extent `pshape[i] *
/// ... * pshape[i+m]` and strides `pstrides[:][i+m]`.
///
/// This simplification has the property that it does not change the iteration
/// order assuming C-order iteration using `pshape` and `pstrides`.
///
/// \param shape The multi-dimensional shape to be iterated over.
/// \param dimension_order Pointer to array of length `shape.size()` that is a
///     permutation of `[0, shape.size())` specifying the iteration order.  Each
///     element in `dimension_order` specifies the corresponding index into
///     `shape` and `strides[j]`.  The iteration will be done in C/row-major
///     order with respect to the specified ordering, meaning that dimension
///     `dimension_order[shape.size()-1]` will change the fastest.
/// \param strides Array of pointers to stride arrays of length `shape.size()`.
///
/// \return The permuted and simplified shape and arrays of strides.
template <std::size_t N>
StridedIterationLayout<N> PermuteAndSimplifyStridedIterationLayout(
    const Index* shape, span<const DimensionIndex> dimension_order,
    std::array<const Index*, N> strides) {
  StridedIterationLayout<N> result;
  const DimensionIndex rank = dimension_order.size();
  if (rank == 0) {
    return result;
  }

  const auto get_size_and_strides =
      [&](DimensionIndex source_dimension_i) -> DimensionSizeAndStrides<N> {
    DimensionSizeAndStrides<N> result;
    result.size = shape[source_dimension_i];
    for (size_t j = 0; j < N; ++j) {
      result.strides[j] = strides[j][source_dimension_i];
    }
    return result;
  };

  const auto can_combine_dimensions =
      [&](const DimensionSizeAndStrides<N>& source,
          const std::array<Index, N>& prior_strides) {
        for (std::size_t j = 0; j < N; ++j) {
          if (source.strides[j] * source.size != prior_strides[j]) return false;
        }
        return true;
      };

  // Add the first input dimension as an output dimension, since we don't yet
  // have anything to combine it with.
  result.push_back(get_size_and_strides(dimension_order[0]));

  for (DimensionIndex i = 1; i < rank; ++i) {
    const DimensionIndex source_dimension_i = dimension_order[i];
    auto source = get_size_and_strides(source_dimension_i);

    // Determine if we can combine `source_dimension_i` with the previous output
    // dimension.
    auto& last_output = result.back();
    if (can_combine_dimensions(source, last_output.strides)) {
      // Combine `source_dimension_i` into the previous output dimension.
      source.size *= last_output.size;
      last_output = source;
      continue;
    }

    // We cannot combine `source_dimension_i` with the previous output
    // dimension, so add it as a new output dimension.
    result.push_back(source);
  }
  return result;
}

/// Computes the ordered list of dimensions over which to iterate.
///
/// \param constraints If `constraints.has_order_constraint()`, constrains the
///     nested iteration order to be either C-order (right to left, last
///     dimension changes fastest) or Fortran-order (left to right, first
///     dimension changes fastest).  Otherwise, an iteration order is chosen by
///     sorting the dimension in order of "decreasing" stride, where this order
///     is determined by lexicographical comparison of `(strides[0][i], ...,
///     strides[strides.size()-1][i])`.  If
///     `constraints.can_skip_repeated_elements()`, dimensions for which all
///     strides are 0 are excluded from iteration.
/// \param shape The shape over which to iterate.
/// \param strides Array of pointers to stride arrays of length `shape.size()`.
/// \returns The dimension order array, containing integers in the range
///     `[0, shape.size())`.
absl::InlinedVector<DimensionIndex, internal::kNumInlinedDims>
ComputeStridedLayoutDimensionIterationOrder(IterationConstraints constraints,
                                            span<const Index> shape,
                                            span<const Index* const> strides);

/// Reorder and simplify a strided iteration layout.
///
/// Refer to the documentation of `ComputeStridedLayoutDimensionIterationOrder`
/// for a description of the parameters.
///
/// \returns The permuted and simplified shape and arrays of strides, intended
///     for iteration in C (right to left) order.
template <std::size_t N>
StridedIterationLayout<N> SimplifyStridedIterationLayout(
    IterationConstraints constraints, span<const Index> shape,
    std::array<const Index*, N> strides) {
  const auto dimension_order =
      ComputeStridedLayoutDimensionIterationOrder(constraints, shape, strides);
  return PermuteAndSimplifyStridedIterationLayout(shape.data(), dimension_order,
                                                  strides);
}

/// Extracts the `InnerRank` innermost dimensions of `iteration_layout`.
///
/// Logically, constructs a `padded_iteration_layout` by padding
/// `iteration_layout` to have rank at least `InnerRank` by prepending singleton
/// dimensions with extent `1` and stride `0`.  Then copies the last `InnerRank`
/// layout dimensions of `padded_iteration_layout` into the return value, and
/// sets `iteration_layout` to the first `padded_iteration_layout.size() -
/// InnerRank` dimensions of `padded_iteration_layout`.
///
/// \tparam InnerRank The number of inner dimensions to extract.
/// \tparam N The number of layouts (stride arrays).
/// \param[in,out] iteration_layout The full iteration layout.  On return, the
///     innermost (i.e. last) `InnerRank` dimensions are removed.  If
///     `InnerRank` is greater than the initial value of
///     `iteration_layout->size()`, `iteration_layout->size()` will become `0`.
/// \return The array `inner_layout` of extracted inner dimension extents and
///     strides.  `inner_layout.shape[i] =
///     padded_iteration_layout[padded_iteration_layout.size() - InnerRank +
///     i].size` if `padded_iteration_layout.size() - InnerRank + i >= 0`, or
///     `0` otherwise.  `inner_layout.strides[j][i] =
///     padded_iteration_layout[padded_iteration_layout.size() - InnerRank +
///     i][j].
template <DimensionIndex InnerRank, std::size_t N>
InnerShapeAndStrides<N, InnerRank> ExtractInnerShapeAndStrides(
    StridedIterationLayout<N>* iteration_layout) {
  InnerShapeAndStrides<N, InnerRank> inner_shapes_and_strides;
  const DimensionIndex full_rank = iteration_layout->size();
  for (DimensionIndex i = 0; i < InnerRank; ++i) {
    DimensionIndex full_dimension_i = full_rank - InnerRank + i;
    if (full_dimension_i >= 0) {
      const auto& layout = (*iteration_layout)[full_dimension_i];
      inner_shapes_and_strides.shape[i] = layout.size;
      for (size_t j = 0; j < N; ++j) {
        inner_shapes_and_strides.strides[j][i] = layout.strides[j];
      }
    } else {
      inner_shapes_and_strides.shape[i] = 1;
      for (size_t j = 0; j < N; ++j) {
        inner_shapes_and_strides.strides[j][i] = 0;
      }
    }
  }
  const DimensionIndex outer_rank =
      InnerRank > full_rank ? 0 : full_rank - InnerRank;
  iteration_layout->resize(outer_rank);
  return inner_shapes_and_strides;
}

/// Helper class that implements the recursive iteration over a
/// multi-dimensional shape.
template <typename Func, typename... Pointer>
class IterateHelper {
 public:
  constexpr static std::size_t arity = sizeof...(Pointer);
  using Result = std::invoke_result_t<Func, Pointer...>;

  /// Calls `func` for each position within the multi-dimensional layout.
  static Result Start(Func func,
                      span<const DimensionSizeAndStrides<arity>> layouts,
                      Pointer... pointers) {
    if (layouts.size() == 0) {
      return func(pointers...);
    }
    return Loop(func, layouts, absl::index_sequence_for<Pointer...>(),
                pointers...);
  }

 private:
  /// Loops over the next dimension, and either recurses or calls `func`.
  ///
  /// \pre layouts.size() >= 1
  template <std::size_t... Is>
  static Result Loop(Func func,
                     span<const DimensionSizeAndStrides<arity>> layouts,
                     absl::index_sequence<Is...> index_sequence,
                     Pointer... pointers) {
    const DimensionSizeAndStrides<arity> size_and_strides = layouts[0];
    auto increment_pointers = [&]() ABSL_ATTRIBUTE_ALWAYS_INLINE {
      auto unused ABSL_ATTRIBUTE_UNUSED = {
          (static_cast<void>(pointers += size_and_strides.strides[Is]), 0)...};
    };
    Result result = DefaultIterationResult<Result>::value();
    if (layouts.size() == 1) {
      for (Index i = 0; i < size_and_strides.size; ++i) {
        result = func(pointers...);
        if (!result) break;
        increment_pointers();
      }
    } else {
      for (Index i = 0; i < size_and_strides.size; ++i) {
        auto result = Loop(func, {&layouts[1], layouts.size() - 1},
                           index_sequence, pointers...);
        if (!result) break;
        increment_pointers();
      }
    }
    return result;
  }
};

#define TENSORSTORE_INTERNAL_FOR_EACH_ARITY(X, ...) \
  X(1)                                              \
  X(2)                                              \
  X(3)                                              \
  X(4)                                              \
  X(5)                                              \
  /**/

}  // namespace internal_iterate
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ITERATE_IMPL_H_
