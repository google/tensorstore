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

#ifndef TENSORSTORE_CONTIGUOUS_LAYOUT_H_
#define TENSORSTORE_CONTIGUOUS_LAYOUT_H_

#include <stddef.h>

#include <cassert>
#include <iosfwd>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Specifies a C-order or Fortran-order contiguous array layout.
///
/// \relates StridedLayout
enum class ContiguousLayoutOrder {
  right = 0,
  c = 0,
  row_major = 0,
  left = 1,
  fortran = 1,
  column_major = 1
};

/// Prints to an `std::ostream`.
/// \relates ContiguousLayoutOrder
///
/// \id ContiguousLayoutOrder
std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order);

/// \relates ContiguousLayoutOrder
constexpr ContiguousLayoutOrder c_order = ContiguousLayoutOrder::c;
constexpr ContiguousLayoutOrder row_major_order =
    ContiguousLayoutOrder::row_major;
constexpr ContiguousLayoutOrder fortran_order = ContiguousLayoutOrder::fortran;
constexpr ContiguousLayoutOrder column_major_order =
    ContiguousLayoutOrder::column_major;

/// Specifies a contiguous layout order as a permutation.
///
/// The first element specifies the outermost dimension and the last element
/// specifies the innermost dimension.
///
/// `ContiguousLayoutOrder::c` corresponds to ``{0, 1, 2, ...}`` while
/// `ContiguousLayoutOrder::fortran` corresponds to ``{..., 2, 1, 0}``.
///
/// \relates StridedLayout
template <DimensionIndex Rank = dynamic_rank>
struct ContiguousLayoutPermutation
    : public tensorstore::span<const DimensionIndex, Rank> {
  /// Constructs a rank-0 (empty) permutation.
  ContiguousLayoutPermutation() = default;

  /// Converts from another `ContiguousLayoutPermutation` with compatible rank.
  ///
  /// \id convert
  template <DimensionIndex R,
            typename =
                std::enable_if_t<R != Rank && RankConstraint::Implies(R, Rank)>>
  ContiguousLayoutPermutation(ContiguousLayoutPermutation<R> other)
      : tensorstore::span<const DimensionIndex, Rank>(other) {}

  /// Constructs from a span.
  ///
  /// \id span
  explicit ContiguousLayoutPermutation(
      tensorstore::span<const DimensionIndex, Rank> permutation)
      : tensorstore::span<const DimensionIndex, Rank>(permutation) {}
};

template <DimensionIndex Rank>
ContiguousLayoutPermutation(
    tensorstore::span<const DimensionIndex, Rank> permutation)
    -> ContiguousLayoutPermutation<Rank>;

template <DimensionIndex Rank>
ContiguousLayoutPermutation(const DimensionIndex (&permutation)[Rank])
    -> ContiguousLayoutPermutation<Rank>;

/// Bool-valued metafunction that is true if `T` is `ContiguousLayoutOrder` or
/// `ContiguousLayoutPermutation` compatible with `Rank`.
///
/// \tparam Rank Rank with which the order must be compatible, or `dynamic_rank`
///     for no constraint.
///
/// \relates StridedLayout
template <typename T, DimensionIndex Rank = dynamic_rank>
constexpr inline bool IsContiguousLayoutOrder = false;

template <DimensionIndex Rank>
constexpr inline bool IsContiguousLayoutOrder<ContiguousLayoutOrder, Rank> =
    true;

template <DimensionIndex R, DimensionIndex Rank>
constexpr inline bool
    IsContiguousLayoutOrder<ContiguousLayoutPermutation<R>, Rank> =
        RankConstraint::Implies(R, Rank);

/// Sets `permutation` to ascending or descending order.
///
/// If `order == c_order`, sets `permutation` to
/// ``{0, 1, ..., permutation.size()-1}``.
///
/// Otherwise, sets `permutation` to ``{permutation.size()-1, ..., 1, 0}``.
///
/// \relates ContiguousLayoutPermutation
void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation);

/// Returns `true` if `permutation` is a valid permutation of
/// ``{0, 1, ..., permutation.size()-1}``.
///
/// \relates ContiguousLayoutPermutation
bool IsValidPermutation(span<const DimensionIndex> permutation);

/// Returns `true` if `permutation` is ``{0, 1, ..., permutation.size()-1}`` if
/// `order == c_order`, or ``{permutation.size() - 1, ..., 1, 0}`` if
/// `order == fortran_order`.
///
/// \relates ContiguousLayoutPermutation
bool PermutationMatchesOrder(span<const DimensionIndex> permutation,
                             ContiguousLayoutOrder order);

/// Sets `inverse_perm` to the inverse permutation of `perm`.
///
/// \param perm[in] Pointer to array of length `rank`.
/// \param inverse_perm[out] Pointer to array of length `rank`.
///
/// \dchecks `IsValidPermutation({perm, rank})`.
/// \relates ContiguousLayoutPermutation
void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm);

/// Normalizes `source` to a permutation if it is not already a permuation.
///
/// \relates ContiguousLayoutPermutation
template <typename LayoutOrder, DimensionIndex Rank>
std::enable_if_t<IsContiguousLayoutOrder<LayoutOrder, Rank>>
ConvertToContiguousLayoutPermutation(LayoutOrder source,
                                     span<DimensionIndex, Rank> target) {
  if constexpr (std::is_same_v<LayoutOrder, ContiguousLayoutOrder>) {
    SetPermutation(source, target);
  } else {
    assert(source.size() == target.size());
    assert(IsValidPermutation(source));
    for (DimensionIndex i = 0; i < target.size(); ++i) {
      target[i] = source[i];
    }
  }
}

/// Computes strides for the specified layout order.
///
/// \param order The layout order.
/// \param element_stride The stride of the innermost dimension.
/// \param shape The extents of the array.
/// \param strides The output strides array.
/// \relates StridedLayout
void ComputeStrides(ContiguousLayoutOrder order, ptrdiff_t element_stride,
                    tensorstore::span<const Index> shape,
                    tensorstore::span<Index> strides);
void ComputeStrides(ContiguousLayoutPermutation<> permutation,
                    ptrdiff_t element_stride,
                    tensorstore::span<const Index> shape,
                    tensorstore::span<Index> strides);

/// Computes the offset of a given index vector in C or Fortran order.
///
/// This is the inverse of `GetContiguousIndices`.
///
/// \param shape Shape for which to compute the offset.
/// \param indices Indices for which to compute the offset.
/// \dchecks `shape.size() == indices.size()`
/// \relates ContiguousLayoutOrder
template <ContiguousLayoutOrder Order = c_order, typename I = Index>
inline I GetContiguousOffset(tensorstore::span<const I> shape,
                             tensorstore::span<const I> indices) {
  assert(shape.size() == indices.size());
  I offset = 0;
  for (ptrdiff_t i = (Order == c_order) ? 0 : (indices.size() - 1);
       (Order == c_order) ? (i < indices.size()) : (i >= 0);
       (Order == c_order) ? ++i : --i) {
    assert(indices[i] >= 0 && indices[i] < shape[i]);
    offset *= shape[i];
    offset += indices[i];
  }
  return offset;
}

/// Computes the indices corresponding to a given linear offset for the
/// specified `shape` and `Order`.
///
/// This is the inverse of `GetContiguousOffset`.
///
/// \param offset Offset for which to the compute the indices.  Must satisfy
///     `0 <= offset && offset < ProductOfExtents(shape)`.
/// \param shape Shape for which to compute the indices.
/// \param indices[out] Indices to be set.
/// \dchecks `shape.size() == indices.size()`
/// \relates ContiguousLayoutOrder
template <ContiguousLayoutOrder Order = c_order, typename I = Index>
inline void GetContiguousIndices(I offset, tensorstore::span<const I> shape,
                                 tensorstore::span<I> indices) {
  assert(shape.size() == indices.size());
  assert(offset >= 0);
  ptrdiff_t rank = shape.size();
  for (ptrdiff_t i = (Order == c_order) ? (rank - 1) : 0;
       (Order == c_order) ? (i >= 0) : (i < rank);
       (Order == c_order) ? --i : ++i) {
    const I size = shape[i];
    indices[i] = offset % size;
    offset /= size;
  }
  assert(offset == 0);
}

/// Sets `permutation` to a permutation that matches the dimension order of
/// `strides`.
///
/// Specifically, `permutation` is ordered by descending stride magnitude,
/// and then ascending dimension index.
///
/// \relates ContiguousLayoutPermutation
void SetPermutationFromStrides(span<const Index> strides,
                               span<DimensionIndex> permutation);

}  // namespace tensorstore

#endif  // TENSORSTORE_CONTIGUOUS_LAYOUT_H_
