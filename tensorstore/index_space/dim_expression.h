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

#ifndef TENSORSTORE_INDEX_SPACE_DIM_EXPRESSION_H_
#define TENSORSTORE_INDEX_SPACE_DIM_EXPRESSION_H_

/// \file
///
/// A DimExpression represents an ordered "selection" of dimensions of an index
/// space and a sequence of "operations" to apply to those dimensions.
///
/// Most users should #include  "third_party/tensorstore/index_space.h" instead.

#include <type_traits>
#include <utility>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/add_new_dims_op.h"
#include "tensorstore/index_space/internal/diagonal_op.h"
#include "tensorstore/index_space/internal/dim_expression_helper.h"
#include "tensorstore/index_space/internal/dimension_selection.h"
#include "tensorstore/index_space/internal/index_array_slice_op.h"
#include "tensorstore/index_space/internal/interval_slice_op.h"
#include "tensorstore/index_space/internal/label_op.h"
#include "tensorstore/index_space/internal/mark_explicit_op.h"
#include "tensorstore/index_space/internal/single_index_slice_op.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/internal/translate_op.h"
#include "tensorstore/index_space/internal/transpose_op.h"

namespace tensorstore {

/// Bool-valued metafunction that evaluates to `true` if `T` is an array type
/// convertible to `SharedArrayView<const Index>`.
///
/// \ingroup indexing
template <typename T>
constexpr inline bool IsIndexArray =
    IsArray<T> && std::is_convertible_v<T, SharedArrayView<const Index>>;

/// A `DimExpression` represents an ordered "selection" of dimensions of an
/// index space and a sequence of "operations" to apply to those dimensions.
///
/// Logically, each operation is a function that maps an index transform and an
/// ordered subset of dimension indices to a new index transform and a new
/// ordered subset of dimension indices.  The returned dimension indices
/// correspond to the dimensions created or modified by the operation, excluding
/// any dimensions that were deleted.
///
/// When a DimExpression is applied to an index transform, the dimensions in the
/// "selection", which may have been specified by label or in other indirect
/// ways, are resolved to an actual ordered subset of dimension indices, and
/// passed along with the index transform as a parameter to the first operation
/// in the sequence.  Subsequent operations, if any, are called using the
/// modified index transform and modified dimension subset returned by the prior
/// operation in the sequence.
///
/// A `DimExpression` with an empty sequence of operations is created by calling
/// `Dims`, `AllDims`, or `DimRange`.  A new `DimExpression` that extends the
/// sequence of operations of an existing `DimExpression` is created by calling
/// member functions, e.g. `IndexSlice`, `IndexArraySlice`, `ClosedInterval`,
/// `TranslateBy`, `TranslateTo`, `MoveTo`, `Label`, `Diagonal`, etc.
///
/// .. warning::
///
///    A `DimExpression` may hold references to temporary values, in which case
///    it must not be used after evaluation of the complete expression in which
///    those temporaries were created.  For example::
///
///        Dims(std::string("x"), std::string("y")).IndexSlice({1, 2})
///
///    refers to data owned by two temporary `std::string` values and a
///    temporary array containing ``{1, 2}``, and it is therefore unsafe to
///    store the resulting `DimExpression` object in a variable that outlives
///    the complete expression in which it was created, as in the following
///    unsafe code::
///
///        auto do_not_use =
///          Dims(std::string("x"), std::string("y")).IndexSlice({1, 2});
///
///
/// The behavior of some operations is specified in terms of an
/// ``interleave`` function.  Given a list ``a`` of length ``n``, a
/// list ``indices`` of ``k`` unique indices in the range
/// ``[0, n + k)``, and a list ``b`` of length ``k``,
/// ``interleave(a, indices, b)`` is defined to be the list ``c`` of
/// length ``length(a) + length(b)`` with ``c[j] = b[i]`` if
/// ``j = indices[i]``, and ``c[j] = a[j - Count(indices < j)]`` if
/// ``j`` not in ``indices``.
///
/// \ingroup indexing
template <typename... Op>
class DimExpression {
  static_assert(sizeof...(Op) > 0);
  using DimExpressionHelper = internal_index_space::DimExpressionHelper;
  using Access = internal_index_space::TransformAccess;
  using Parent =
      typename internal_index_space::DimExpressionTraits<Op...>::Parent;
  using LastOp =
      typename internal_index_space::DimExpressionTraits<Op...>::LastOp;

  // The static rank of the dimension selection, as determined without knowing
  // the static input rank of the index transform to which this DimExpression
  // will be applied.
  //
  // This is defined as an integral_constant type rather than a static
  // constexpr member so that we can use it to specify return types of member
  // functions.
  using static_selection_rank =
      std::integral_constant<DimensionIndex,
                             DimExpressionHelper::GetStaticSelectionRank<Op...>(
                                 dynamic_rank)>;

  // Type alias for a DimExpression that chains NextOp to the end of this
  // DimExpression.
  template <typename NextOp>
  using NewExpr = DimExpression<NextOp, Op...>;

  // Type alias for a DimExpression that chains OpTemplate<IndexVector...> to
  // the end of this DimExpression.
  //
  // \requires Each `IndexVector` satisfies the IsIndexVectorOrScalar concept
  //     with compatible static ranks, that are compatible with the static
  //     selection rank.
  template <template <typename...> class OpTemplate, typename... IndexVector>
  using IndexVectorOpExpr = NewExpr<DimExpressionHelper::IndexVectorOp<
      OpTemplate, static_selection_rank::value, IndexVector...>>;

  // Defines the return type for TranslateBy.
  template <typename IndexVector>
  using TranslateByOpExpr =
      IndexVectorOpExpr<internal_index_space::TranslateByOp, IndexVector>;

  // Defines the return type for TranslateBackwardBy.
  template <typename IndexVector>
  using TranslateBackwardByOpExpr =
      IndexVectorOpExpr<internal_index_space::TranslateBackwardByOp,
                        IndexVector>;

  // Defines the return type for TranslateTo.
  template <typename IndexVector>
  using TranslateToOpExpr =
      IndexVectorOpExpr<internal_index_space::TranslateToOp, IndexVector>;

  // Defines the return type for Stride.
  template <typename IndexVector>
  using StrideOpExpr =
      IndexVectorOpExpr<internal_index_space::StrideOp, IndexVector>;

  // Defines the return type for IndexSlice with an index vector.
  template <typename IndexVector>
  using SingleIndexSliceOpExpr =
      IndexVectorOpExpr<internal_index_space::SingleIndexSliceOp, IndexVector>;

  // Defines the return type for the *Interval member functions.
  template <typename... IndexVector>
  using IntervalSliceOpExpr =
      IndexVectorOpExpr<internal_index_space::IntervalSliceOp, IndexVector...>;

  template <typename BoxType>
  using BoxSliceOpExpr = NewExpr<std::enable_if_t<
      (IsBoxLike<BoxType> &&
       RankConstraint::EqualOrUnspecified(static_selection_rank::value,
                                          BoxType::static_rank)),
      internal_index_space::BoxSliceOp<BoxType::static_rank>>>;

  // Defines the return type for IndexArraySlice with a parameter pack of index
  // arrays.
  template <typename... IndexArray>
  using IndexArraySliceOpExpr = std::enable_if_t<
      (sizeof...(IndexArray) >= 1) &&
          RankConstraint::EqualOrUnspecified(sizeof...(IndexArray),
                                             static_selection_rank::value) &&
          (IsIndexArray<IndexArray> && ...) &&
          RankConstraint::EqualOrUnspecified({IndexArray::static_rank...}),
      NewExpr<internal_index_space::IndexArraySliceOp<
          /*OuterIndexing=*/false,
          RankConstraint::And({IndexArray::static_rank...}),
          std::array<SharedArrayView<const Index>, sizeof...(IndexArray)>>>>;

  // Defines the return type for IndexArraySlice with a span of index arrays.
  using DynamicIndexArraySliceOpExpr =
      NewExpr<internal_index_space::IndexArraySliceOp<
          /*OuterIndexing=*/false, dynamic_rank,
          span<const SharedArrayView<const Index>>>>;

  // Defines the return type for OuterIndexArraySlice with a parameter pack of
  // index arrays.
  template <typename... IndexArray>
  using IndexArrayOuterSliceOpExpr = std::enable_if_t<
      RankConstraint::EqualOrUnspecified(sizeof...(IndexArray),
                                         static_selection_rank::value) &&
          (IsIndexArray<IndexArray> && ...),
      NewExpr<internal_index_space::IndexArraySliceOp<
          /*OuterIndexing=*/true,
          RankConstraint::Add({IndexArray::static_rank...}),
          std::array<SharedArrayView<const Index>, sizeof...(IndexArray)>>>>;

  // Defines the return type for OuterIndexArraySlice with a span of index
  // arrays.
  using DynamicIndexArrayOuterSliceOpExpr =
      NewExpr<internal_index_space::IndexArraySliceOp<
          /*OuterIndexing=*/true, dynamic_rank,
          span<const SharedArrayView<const Index>>>>;

  // Defines the return type for Label using the specified `Labels` container
  // with the specified static `Rank`.
  template <typename Labels, DimensionIndex Rank>
  using LabelOpExpr =
      std::enable_if_t<RankConstraint::EqualOrUnspecified(
                           Rank, static_selection_rank::value),
                       NewExpr<internal_index_space::LabelOp<Labels>>>;

  // Defines the return type for Label, where the `Labels` container is
  // converted to a `span`.
  template <typename Labels,
            typename LabelsSpan = internal::ConstSpanType<Labels>>
  using LabelSpanOpExpr =
      std::enable_if_t<internal::IsStringLike<typename LabelsSpan::value_type>,
                       LabelOpExpr<LabelsSpan, LabelsSpan::extent>>;

  // Defines the return type for Label, where the labels are specified as an
  // argument pack.
  template <typename... Label>
  using LabelPackOpExpr = std::enable_if_t<
      internal::IsPackConvertibleWithoutNarrowing<std::string_view, Label...>,
      LabelOpExpr<std::array<std::string_view, sizeof...(Label)>,
                  sizeof...(Label)>>;

  // Defines the return type for MoveTo, MoveToFront, and MoveToBack.
  using MoveToOpExpr = NewExpr<internal_index_space::MoveToOp>;

  // Defines the return type for Diagonal.
  using DiagonalOpExpr = NewExpr<internal_index_space::DiagonalOp>;

  // Defines the return type for AddNew.
  using AddNewOpExpr = NewExpr<internal_index_space::AddNewDimsOp>;

  // Defines the return type for `Transpose()`.
  using TransposeOpExpr = NewExpr<internal_index_space::TransposeOp>;

  // Defines the return type for `Transpose(target_dimensions)`.
  template <typename TargetDims,
            typename TargetDimsSpan = internal::ConstSpanType<TargetDims>>
  using TransposeToOpExpr = std::enable_if_t<
      (RankConstraint::EqualOrUnspecified(TargetDimsSpan::extent,
                                          static_selection_rank::value) &&
       std::is_same_v<typename TargetDimsSpan::value_type, DimensionIndex>),
      NewExpr<internal_index_space::TransposeToOp<TargetDimsSpan>>>;

  // Defines the return type for {Unsafe,}MarkBounds{Explicit,Implicit}.
  using ChangeImplicitStateOpExpr =
      NewExpr<internal_index_space::ChangeImplicitStateOp>;

  // Defines the return type for IndexVectorArraySlice.
  template <typename IndexVectorArray>
  using IndexVectorArraySliceOpExpr =
      std::enable_if_t<IsIndexArray<IndexVectorArray> &&
                           RankConstraint::GreaterOrUnspecified(
                               IndexVectorArray::static_rank, 0),
                       NewExpr<internal_index_space::IndexVectorArraySliceOp<
                           IndexVectorArray::static_rank>>>;

 public:
  /// Translates (shifts) the domains of the selected input dimensions by the
  /// specified `offsets` vector; the output range remains the same.
  ///
  /// Given an ``existing`` transform with input rank ``m`` and the
  /// selected ``dims`` vector, the new index transform maps an ``input``
  /// vector of size ``m`` to::
  ///
  ///     existing(input - full_offsets)
  ///
  /// where ``full_offsets`` is a vector of size ``m`` with
  /// ``full_offsets[i] = offsets[j]`` if ``dims[j] == i``, and
  /// ``full_offsets[i] = 0`` if ``i`` is not in ``dims``.  An offset
  /// of `kImplicit` is treated as `0`.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extent of the `offsets` vector.
  ///
  /// The input domain for each selected dimension is shifted by calling
  /// ShiftInterval.
  ///
  /// For example: `Dims(0, 2).TranslateBy({10, 20})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[11, 13], [2, 5], [23, 24]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 3}``
  ///      - ``{12, 3, 23}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{x + 10, y, z + 20}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \requires `Offsets` satisfies the `IsIndexVectorOrScalar` concept with a
  ///     static extent compatible with the static rank of the dimension
  ///     selection.
  /// \param offsets The offset vector by which to shift the input domains of
  ///     the selected dimensions.  May be a braced list,
  ///     e.g. `TranslateBy({1, 2, 3})`.  May also be a scalar,
  ///     e.g. `TranslateBy(5)`, in which case the same offset is used for all
  ///     selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `offsets`
  ///     vector is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kOutOfRange` if the shift offset is outside
  ///     ``[kMinFiniteIndex, kMaxFiniteIndex]``.
  /// \error `absl::StatusCode::kInvalidArgument` if a shifted interval is
  ///     outside the valid range.
  template <typename Offsets>
  TranslateByOpExpr<Offsets> TranslateBy(const Offsets& offsets) const {
    return {{offsets}, *this};
  }

  // Overload that permits the offset vector to be specified as a braced list.
  template <DimensionIndex Rank>
  TranslateByOpExpr<const Index (&)[Rank]> TranslateBy(
      const Index (&offsets)[Rank]) const {
    return {{span(offsets)}, *this};
  }

  /// Translates (shifts) the domains of the selected input dimensions backwards
  /// by the specified `offsets` vector; the output range remains the same.
  ///
  /// Given an ``existing`` transform with input rank ``m`` and the
  /// selected ``dims`` vector, the new index transform maps an ``input``
  /// vector of size ``m`` to::
  ///
  ///     existing(input + full_offsets)
  ///
  /// where ``full_offsets`` is a vector of size ``m`` with
  /// ``full_offsets[i] = offsets[j]`` if ``dims[j] == i``, and
  /// ``full_offsets[i] = 0`` if ``i`` is not in ``dims``.  An offset
  /// of `kImplicit` is treated as `0`.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extent of the `offsets` vector.
  ///
  /// The input domain for each selected dimension is shifted by calling
  /// `ShiftIntervalBackward`.
  ///
  /// For example: `Dims(0, 2).TranslateBackwardBy({10, 20})` has the following
  /// effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[-9, -7], [2, 5], [-17,-16]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 3}``
  ///      - ``{-8, 3, -17}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{x - 10, y, z - 20}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \requires `Offsets` satisfies the `IsIndexVectorOrScalar` concept with a
  ///     static extent compatible with the static rank of the dimension
  ///     selection.
  /// \param offsets The offset vector by which to shift the input domains of
  ///     the selected dimensions.  May be a braced list,
  ///     e.g. `TranslateBackwardBy({1, 2, 3})`.  May also be a scalar,
  ///     e.g. `TranslateBackwardBy(5)`, in which case the same offset is used
  ///     for all selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `offsets`
  ///     vector is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kOutOfRange` if the shift offset is outside
  ///     ``[kMinFiniteIndex, kMaxFiniteIndex]``.
  /// \error `absl::StatusCode::kInvalidArgument` if a shifted interval is
  ///     outside the valid range.
  template <typename Offsets>
  TranslateBackwardByOpExpr<Offsets> TranslateBackwardBy(
      const Offsets& offsets) const {
    return {{offsets}, *this};
  }

  // Overload that permits the offset vector to be specified as a braced list.
  template <DimensionIndex Rank>
  TranslateBackwardByOpExpr<const Index (&)[Rank]> TranslateBackwardBy(
      const Index (&offsets)[Rank]) const {
    return {{span(offsets)}, *this};
  }

  /// Translates the domain of the selected input dimensions to the specified
  /// origin vector without affecting the output range.
  ///
  /// Given an ``existing`` transform with input rank ``m`` and the
  /// selected ``dims`` vector, the new index transform maps an ``input``
  /// vector of size ``m`` to::
  ///
  ///     existing(input - full_offsets)
  ///
  /// where ``full_offsets`` is a vector of size ``m`` with
  /// ``full_offsets[i] = origins[j] - existing.input_origin(i)`` if
  /// ``dims[j] = i``, and ``full_offsets[i] = 0`` if ``i`` is not in
  /// ``dims``.  As a special case, an origin of `kImplicit` specifies no
  /// translation of the corresponding dimension.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extent of the `origins` vector.
  ///
  /// For example: `Dims(0, 2).TranslateTo({10, 20})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[10, 12], [2, 5], [20, 21]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 3}``
  ///      - ``{11, 3, 20}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{x + 9, y, z + 17}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \requires `Origins` satisfies the IsIndexVectorOrScalar concept with a
  ///     static extent compatible with the static rank of the dimension
  ///     selection.
  /// \param origins The origin vector to which to shift the input domains of
  ///     the selected dimensions.  May be a braced list,
  ///     e.g. `TranslateTo({1, 2, 3})`.  May also be a scalar,
  ///     e.g. `TranslateTo(5)`, in which case the same origin is used for all
  ///     selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `origins`
  ///     vector is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the input domain of any
  ///     selected dimension has an `IndexInterval::inclusive_min` value of
  ///     `-kInfIndex`.
  /// \error `absl::StatusCode::kOutOfRange` if any origin value is outside
  ///     ``[kMinFiniteIndex, kMaxFiniteIndex]``.
  /// \error `absl::StatusCode::kInvalidArgument` if a shifted interval is
  ///     outside the valid range.
  template <typename Origins>
  TranslateToOpExpr<Origins> TranslateTo(const Origins& origins) const {
    return {{origins}, *this};
  }

  // Overload that permits the origin vector to be specified as a braced list.
  template <DimensionIndex Rank>
  TranslateToOpExpr<const Index (&)[Rank]> TranslateTo(
      const Index (&origins)[Rank]) const {
    return {{span(origins)}, *this};
  }

  /// Extracts a single-index slice of the selected dimensions using the
  /// specified index vector.
  ///
  /// Given ``k`` ``selected_dimensions``, an ``indices`` vector of
  /// size ``k``, and an ``existing_transform`` with input rank ``m``,
  /// the new index transform maps ``input`` vectors of size ``m - k``
  /// to::
  ///
  ///     existing_transform(interleave(input, selected_dimensions, indices))
  ///
  /// The selected dimensions are removed from the new index space, and the new
  /// dimension selection is empty.  Dimensions that are not selected are
  /// retained.
  ///
  /// For example: `Dims(0, 2).IndexSlice({2, 4})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[2, 5]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"y"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 4}``
  ///      - ``{3}``
  ///    * - Equivalent input indices
  ///      - ``{2, y, 4}``
  ///      - ``{y}``
  ///
  /// where ``y`` is any index in ``[2, 5]``.
  ///
  /// \requires `Indices` satisfies the IsIndexVectorOrScalar concept with a
  ///     static extent compatible with the static rank of the dimension
  ///     selection.
  /// \param indices The index vector specifying the index to slice from each
  ///     selected dimension.  May be a braced list, e.g.
  ///     `IndexSlice({1, 2, 3})`. May also be a scalar, e.g. `IndexSlice(5)`,
  ///     in which case the same index is used for all selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `indices`
  ///     vector is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kOutOfRange` if the index value for a given
  ///     input dimension is outside its effective domain (implicit lower/upper
  ///     bounds are treated as -/+inf).
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
  template <typename Indices>
  SingleIndexSliceOpExpr<Indices> IndexSlice(const Indices& indices) const {
    return {{indices}, *this};
  }

  // Overload that permits the indices vector to be specified as a braced list.
  template <DimensionIndex Rank>
  SingleIndexSliceOpExpr<const Index (&)[Rank]> IndexSlice(
      const Index (&indices)[Rank]) const {
    return {{span(indices)}, *this};
  }

  /// Extracts a box from the selected dimensions.
  ///
  /// This is equivalent to `SizedInterval(box.origin(), box.shape())`.
  ///
  /// For example: `Dims(0, 2).BoxSlice(BoxView({1, 4}, {3, 4}))` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 3], [2, 5], [4, 7]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{1, 3, 4}``
  ///      - ``{1, 3, 4}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[4, 7]``.
  ///
  /// \requires `BoxType` satisfies `IsBoxLike` and has a rank compatible with
  ///     the static rank of the dimension selection.
  /// \param box The box to extract.
  template <typename BoxType>
  BoxSliceOpExpr<BoxType> BoxSlice(const BoxType& box) const {
    return {{box, false}, *this};
  }

  /// Extracts a box from the selected dimensions, and translates its origin to
  /// 0.
  ///
  /// This is equivalent to `TranslateSizedInterval(box.origin(), box.shape())`.
  ///
  /// For example: `Dims(0, 2).TranslateBoxSlice(BoxView({1, 4}, {3, 4}))` has
  /// the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[0, 2], [2, 5], [0, 3]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{1, 3, 2}``
  ///      - ``{0, 3, 0}``
  ///    * - Equivalent input indices
  ///      - ``{x + 1, y, z + 4}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[0, 2]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[0, 3]``.
  ///
  /// \requires `BoxType` satisfies `IsBoxLike` and has a rank compatible with
  ///     the static rank of the dimension selection.
  /// \param box The box to extract.
  template <typename BoxType>
  BoxSliceOpExpr<BoxType> TranslateBoxSlice(const BoxType& box) const {
    return {{box, true}, *this};
  }

  // [BEGIN GENERATED: generate_interval_slice_overloads.py]
  // The following code is automatically generated.  Do not modify directly.

  /// Extracts a closed interval from the selected dimensions with optional
  /// striding.
  ///
  /// The domain of each selected dimension is transformed by
  /// ExtractClosedStridedSlice using the corresponding components of the
  /// `start`, `stop`, and `strides` vectors.  In the simple case that the
  /// stride component is `1`, the new domain is simply restricted to the
  /// specified interval, with the new origin equal to the specified `start`
  /// component. In the general case with a stide component not equal to `1`,
  /// the new origin is equal to the `start` component divided by the `strides`
  /// component, rounded towards zero; in this case, the
  /// `TranslateClosedInterval` operation, which ensures an origin of 0, may be
  /// more convenient.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extents of the `start`, `stop`, and `strides`
  /// vectors.
  ///
  /// For example: `Dims(0, 2).ClosedInterval({1, 8}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 4], [2, 5], [-4, -2]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -2]``.
  ///
  /// Note that in the case of a stride component not equal to `1` or `-1`, if
  /// the `start` component is not evenly divisible by the stride, the
  /// transformation involves an additional offset.
  ///
  /// For example: `Dims(0, 2).ClosedInterval({1, 9}, {4, 3}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 4], [2, 5], [-4, -1]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 7}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2 + 1}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -1]``.
  ///
  /// \requires `Start`, `Stop`, and `Strides` satisfy the IsIndexVectorOrScalar
  ///     concept with static extents compatible with each other and with the
  ///     static rank of the dimension selection.
  /// \param start The index vector specifying the start indices for each
  ///     selected dimension.  May be a braced list, e.g. ``{1, 2, 3}``.
  ///     May also be a scalar, e.g. `5`, in which case the same start index is
  ///     used for all selected dimensions.
  /// \param stop The index vector specifying the stop indices for each selected
  ///     dimension.  May be a braced list or scalar.
  /// \param strides The index vector specifying the stride value for each
  ///     selected dimension.  May be a braced list or scalar.  If not
  ///     specified, defaults to 1.
  /// \error `absl::StatusCode::kInvalidArgument` if the extents of the `start`,
  ///     `stop`, or `strides` vectors do not match the number of selected
  ///     dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` or
  ///     `absl::StatusCode::kOutOfRange` if the `start`, `stop`, and
  ///     `strides` values are invalid or specify a slice outside the effective
  ///     bounds for a given dimension (implicit lower/upper bounds are treated
  ///     as -/+inf).
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
  template <typename Start, typename Stop, typename Strides = Index>
  IntervalSliceOpExpr<Start, Stop, Strides> ClosedInterval(
      const Start& start, const Stop& stop, const Strides& strides = 1) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, Strides> ClosedInterval(
      const Index (&start)[Rank], const Stop& stop,
      const Strides& strides = 1) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides> ClosedInterval(
      const Start& start, const Index (&stop)[Rank],
      const Strides& strides = 1) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  ClosedInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                 const Strides& strides = 1) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Stop, const Index (&)[Rank]> ClosedInterval(
      const Start& start, const Stop& stop,
      const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, const Index (&)[Rank]>
  ClosedInterval(const Index (&start)[Rank], const Stop& stop,
                 const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  ClosedInterval(const Start& start, const Index (&stop)[Rank],
                 const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  ClosedInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                 const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, false, start, stop, strides}, *this};
  }

  /// Extracts a half-open interval from the selected dimensions with optional
  /// striding.
  ///
  /// The domain of each selected dimension is transformed by
  /// ExtractHalfOpenStridedSlice using the corresponding components of the
  /// `start`, `stop`, and `strides` vectors.  In the simple case that the
  /// stride component is `1`, the new domain is simply restricted to the
  /// specified interval, with the new origin equal to the specified `start`
  /// component. In the general case with a stide component not equal to `1`,
  /// the new origin is equal to the `start` component divided by the `strides`
  /// component, rounded towards zero; in this case, the
  /// `TranslateHalfOpenInterval` operation, which ensures an origin of 0, may
  /// be more convenient.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extents of the `start`, `stop`, and `strides`
  /// vectors.
  ///
  /// For example: `Dims(0, 2).HalfOpenInterval({1, 8}, {4, 3}, {1, -2})` has
  /// the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 3], [2, 5], [-4, -2]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 4]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -2]``.
  ///
  /// \requires `Start`, `Stop`, and `Strides` satisfy the IsIndexVectorOrScalar
  ///     concept with static extents compatible with each other and with the
  ///     static rank of the dimension selection.
  /// \param start The index vector specifying the start indices for each
  ///     selected dimension.  May be a braced list, e.g. ``{1, 2, 3}``.
  ///     May also be a scalar, e.g. `5`, in which case the same start index is
  ///     used for all selected dimensions.
  /// \param stop The index vector specifying the stop indices for each selected
  ///     dimension.  May be a braced list or scalar.
  /// \param strides The index vector specifying the stride value for each
  ///     selected dimension.  May be a braced list or scalar.  If not
  ///     specified, defaults to 1.
  /// \error `absl::StatusCode::kInvalidArgument` if the extents of the `start`,
  ///     `stop`, or `strides` vectors do not match the number of selected
  ///     dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` or
  ///     `absl::StatusCode::kOutOfRange` if the `start`, `stop`, and
  ///     `strides` values are invalid or specify a slice outside the effective
  ///     bounds for a given dimension (implicit lower/upper bounds are treated
  ///     as -/+inf).
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
  template <typename Start, typename Stop, typename Strides = Index>
  IntervalSliceOpExpr<Start, Stop, Strides> HalfOpenInterval(
      const Start& start, const Stop& stop, const Strides& strides = 1) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, Strides> HalfOpenInterval(
      const Index (&start)[Rank], const Stop& stop,
      const Strides& strides = 1) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides> HalfOpenInterval(
      const Start& start, const Index (&stop)[Rank],
      const Strides& strides = 1) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  HalfOpenInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                   const Strides& strides = 1) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Stop, const Index (&)[Rank]> HalfOpenInterval(
      const Start& start, const Stop& stop,
      const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, const Index (&)[Rank]>
  HalfOpenInterval(const Index (&start)[Rank], const Stop& stop,
                   const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  HalfOpenInterval(const Start& start, const Index (&stop)[Rank],
                   const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  HalfOpenInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                   const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, false, start, stop, strides}, *this};
  }

  /// Extracts a sized interval from the selected dimensions with optional
  /// striding.
  ///
  /// The domain of each selected dimension is transformed by
  /// ExtractSizedStridedSlice using the corresponding components of the
  /// `start`, `size`, and `strides` vectors.  In the simple case that the
  /// stride component is `1`, the new domain is simply restricted to the
  /// specified interval, with the new origin equal to the specified `start`
  /// component. In the general case with a stide component not equal to `1`,
  /// the new origin is equal to the `start` component divided by the `strides`
  /// component, rounded towards zero; in this case, the
  /// `TranslateSizedInterval` operation, which ensures an origin of 0, may be
  /// more convenient.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extents of the `start`, `size`, and `strides`
  /// vectors.
  ///
  /// For example: `Dims(0, 2).SizedInterval({1, 8}, {3, 2}, {1, -2})` has the
  /// following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[1, 3], [2, 5], [-4, -3]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 6}``
  ///      - ``{2, 3, -3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z * -2}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[-4, -3]``.
  ///
  /// \requires `Start`, `Size`, and `Strides` satisfy the IsIndexVectorOrScalar
  ///     concept with static extents compatible with each other and with the
  ///     static rank of the dimension selection.
  /// \param start The index vector specifying the start indices for each
  ///     selected dimension.  May be a braced list, e.g. ``{1, 2, 3}``.
  ///     May also be a scalar, e.g. `5`, in which case the same start index is
  ///     used for all selected dimensions.
  /// \param size The size vector specifying the size of the domain for each
  ///     selected dimension.  May be a braced list or scalar.
  /// \param strides The index vector specifying the stride value for each
  ///     selected dimension.  May be a braced list or scalar.  If not
  ///     specified, defaults to 1.
  /// \error `absl::StatusCode::kInvalidArgument` if the extents of the `start`,
  ///     `size`, or `strides` vectors do not match the number of selected
  ///     dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` or
  ///     `absl::StatusCode::kOutOfRange` if the `start`, `size`, and
  ///     `strides` values are invalid or specify a slice outside the effective
  ///     bounds for a given dimension (implicit lower/upper bounds are treated
  ///     as -/+inf).
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
  template <typename Start, typename Size, typename Strides = Index>
  IntervalSliceOpExpr<Start, Size, Strides> SizedInterval(
      const Start& start, const Size& size, const Strides& strides = 1) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Size, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Size, Strides> SizedInterval(
      const Index (&start)[Rank], const Size& size,
      const Strides& strides = 1) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides> SizedInterval(
      const Start& start, const Index (&size)[Rank],
      const Strides& strides = 1) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  SizedInterval(const Index (&start)[Rank], const Index (&size)[Rank],
                const Strides& strides = 1) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Size, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Size, const Index (&)[Rank]> SizedInterval(
      const Start& start, const Size& size,
      const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Size, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Size, const Index (&)[Rank]>
  SizedInterval(const Index (&start)[Rank], const Size& size,
                const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  SizedInterval(const Start& start, const Index (&size)[Rank],
                const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  SizedInterval(const Index (&start)[Rank], const Index (&size)[Rank],
                const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, false, start, size, strides}, *this};
  }

  /// Equivalent to `ClosedInterval(start, stop, strides).TranslateTo(0)`.
  template <typename Start, typename Stop, typename Strides = Index>
  IntervalSliceOpExpr<Start, Stop, Strides> TranslateClosedInterval(
      const Start& start, const Stop& stop, const Strides& strides = 1) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, Strides>
  TranslateClosedInterval(const Index (&start)[Rank], const Stop& stop,
                          const Strides& strides = 1) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides>
  TranslateClosedInterval(const Start& start, const Index (&stop)[Rank],
                          const Strides& strides = 1) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  TranslateClosedInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                          const Strides& strides = 1) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Stop, const Index (&)[Rank]>
  TranslateClosedInterval(const Start& start, const Stop& stop,
                          const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, const Index (&)[Rank]>
  TranslateClosedInterval(const Index (&start)[Rank], const Stop& stop,
                          const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  TranslateClosedInterval(const Start& start, const Index (&stop)[Rank],
                          const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  TranslateClosedInterval(const Index (&start)[Rank], const Index (&stop)[Rank],
                          const Index (&strides)[Rank]) const {
    return {{IntervalForm::closed, true, start, stop, strides}, *this};
  }

  /// Equivalent to `HalfOpenInterval(start, stop, strides).TranslateTo(0)`.
  template <typename Start, typename Stop, typename Strides = Index>
  IntervalSliceOpExpr<Start, Stop, Strides> TranslateHalfOpenInterval(
      const Start& start, const Stop& stop, const Strides& strides = 1) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, Strides>
  TranslateHalfOpenInterval(const Index (&start)[Rank], const Stop& stop,
                            const Strides& strides = 1) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides>
  TranslateHalfOpenInterval(const Start& start, const Index (&stop)[Rank],
                            const Strides& strides = 1) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  TranslateHalfOpenInterval(const Index (&start)[Rank],
                            const Index (&stop)[Rank],
                            const Strides& strides = 1) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Stop, const Index (&)[Rank]>
  TranslateHalfOpenInterval(const Start& start, const Stop& stop,
                            const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Stop, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Stop, const Index (&)[Rank]>
  TranslateHalfOpenInterval(const Index (&start)[Rank], const Stop& stop,
                            const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  TranslateHalfOpenInterval(const Start& start, const Index (&stop)[Rank],
                            const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  TranslateHalfOpenInterval(const Index (&start)[Rank],
                            const Index (&stop)[Rank],
                            const Index (&strides)[Rank]) const {
    return {{IntervalForm::half_open, true, start, stop, strides}, *this};
  }

  /// Equivalent to `SizedInterval(start, size, strides).TranslateTo(0)`.
  template <typename Start, typename Size, typename Strides = Index>
  IntervalSliceOpExpr<Start, Size, Strides> TranslateSizedInterval(
      const Start& start, const Size& size, const Strides& strides = 1) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Size, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Size, Strides>
  TranslateSizedInterval(const Index (&start)[Rank], const Size& size,
                         const Strides& strides = 1) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], Strides>
  TranslateSizedInterval(const Start& start, const Index (&size)[Rank],
                         const Strides& strides = 1) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Strides = Index, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank], Strides>
  TranslateSizedInterval(const Index (&start)[Rank], const Index (&size)[Rank],
                         const Strides& strides = 1) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, typename Size, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, Size, const Index (&)[Rank]>
  TranslateSizedInterval(const Start& start, const Size& size,
                         const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Size, DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], Size, const Index (&)[Rank]>
  TranslateSizedInterval(const Index (&start)[Rank], const Size& size,
                         const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <typename Start, DimensionIndex Rank>
  IntervalSliceOpExpr<Start, const Index (&)[Rank], const Index (&)[Rank]>
  TranslateSizedInterval(const Start& start, const Index (&size)[Rank],
                         const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }

  // Overload that permits arguments to be specified as braced lists.
  template <DimensionIndex Rank>
  IntervalSliceOpExpr<const Index (&)[Rank], const Index (&)[Rank],
                      const Index (&)[Rank]>
  TranslateSizedInterval(const Index (&start)[Rank], const Index (&size)[Rank],
                         const Index (&strides)[Rank]) const {
    return {{IntervalForm::sized, true, start, size, strides}, *this};
  }
  // [END GENERATED: generate_interval_slice_overloads.py]

  /// Jointly slices the selected dimensions using index arrays.
  ///
  /// The ``k = sizeof...(IndexArray)`` index arrays, corresponding to the
  /// ``k`` ``selected_dimensions``, must all be of the same rank ``n``
  /// and have broadcast-compatible shapes.  Given an ``existing_transform``
  /// with input rank ``m``, the new index transform maps ``input``
  /// vectors of size ``m + n - k`` to::
  ///
  ///     existing_transform(interleave(input[n:], selected_dimensions,
  ///                                   {broadcast(index_array)(input[:n])...}))
  ///
  /// The selected dimensions are removed from the new index space, and the
  /// dimensions of the index arrays (which must all have the same rank and
  /// broadcast-compatible shapes) are added as the first dimensions of the new
  /// index space with origins of 0 and empty labels.  The new dimension
  /// selection corresponds to these added dimensions; they can be labeled and
  /// reordered relative to the existing dimensions by chaining the Label and
  /// MoveTo methods.
  ///
  /// The behavior is similar to that of NumPy integer array indexing, except
  /// that the added dimensions corresponding to the index arrays are always
  /// added as the first dimensions.
  ///
  /// For example::
  ///
  ///     Dims(0, 2).IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
  ///                                MakeArray<Index>({{7, 8, 9}, {0, 1, 2}}))
  ///
  /// has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 1}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[0, 1], [0, 2], [2, 5]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"", "", "y"}``
  ///    * - Equivalent input indices
  ///      - ``{1, y, 7}``
  ///      - ``{0, 0, y}``
  ///    * - Equivalent input indices
  ///      - ``{2, y, 8}``
  ///      - ``{0, 1, y}``
  ///    * - Equivalent input indices
  ///      - ``{3, y, 9}``
  ///      - ``{0, 2, y}``
  ///    * - Equivalent input indices
  ///      - ``{6, y, 2}``
  ///      - ``{1, 2, y}``
  ///    * - Equivalent input indices
  ///      - ``{xi(a, b), y, zi(a, b)}``
  ///      - ``{a, b, y}``
  ///
  /// where ``y`` is any index in ``[2, 5]``, ``a`` is any index in
  /// ``[0, 1]``, ``b`` is any index in ``[0, 2]``,
  /// ``xi = MakeArray<Index>({{1, 2, 3}, {4, 5, 6}})`` and
  /// ``zi = MakeArray<Index>({{7, 8, 9}, {0, 1, 2}})``.
  ///
  /// \requires The `IndexArray` types satisfy IsIndexArray and have compatible
  ///     static ranks.
  /// \requires For the variadic overload, `sizeof...(IndexArray)` must be
  ///     compatible with the static rank of the dimension selection.
  /// \param index_arrays The index arrays used to index into each selected
  ///     dimension.  The new transform may share ownership of the supplied
  ///     index arrays.  The index arrays should not be modified after being
  ///     passed to this function.  The index values contained in the index
  ///     arrays may be bounds-checked lazily.
  /// \error `absl::StatusCode::kInvalidArgument` if `sizeof...(IndexArray)` is
  ///     not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the shapes of the index
  ///     arrays cannot be broadcast to a common shape.
  /// \error `absl::StatusCode::kOutOfRange` if an out-of-bounds index is
  ///     detected. \error `absl::StatusCode::kInvalidArgument` if integer
  ///     overflow occurs when computing the resultant transform.
  template <typename... IndexArray>
  IndexArraySliceOpExpr<IndexArray...> IndexArraySlice(
      const IndexArray&... index_arrays) const {
    return {{index_arrays...}, *this};
  }
  DynamicIndexArraySliceOpExpr IndexArraySlice(
      span<const SharedArrayView<const Index>> index_arrays) const {
    return {{index_arrays}, *this};
  }

  /// Jointly slices the selected dimensions using the specified array of index
  /// vectors.
  ///
  /// Given ``k`` ``selected_dimensions``, an ``index_vector_array`` of
  /// rank ``n + 1`` with
  /// ``index_vector_array.size(vector_dimension) == k``, and an
  /// ``existing_transform`` with input rank ``m``, the new index
  /// transform maps ``input`` vectors of size ``m + n - k`` to::
  ///
  ///     existing_transform(
  ///         interleave(
  ///             input[n:], selected_dimensions,
  ///             {index_vector_array(
  ///                  interleave(input[:n], {vector_dimension}, i))
  ///              for 0 <= i < k}))
  ///
  /// The selected dimensions are removed from the new index space, and the
  /// dimensions of the index vector array, other than `vector_dimension`, are
  /// added as the first dimensions of the new index space with origins of 0 and
  /// empty labels.  The new dimension selection corresponds to these added
  /// dimensions; they can be labeled and reordered relative to the existing
  /// dimensions by chaining the Label and MoveTo methods.
  ///
  /// In cases where the indices are already arranged as an array of index
  /// vectors, this method provides a more convenient interface than the more
  /// general IndexArraySlice method (which requires separate arrays to be
  /// specified for each selected dimension).
  ///
  /// For example::
  ///
  ///     Dims(0, 2).IndexVectorArraySlice(
  ///         MakeArray<Index>({{{1, 7}, {2, 8}, {3, 9}},
  ///                           {{4, 0}, {5, 1}, {6, 2}}}),
  ///         -1)
  ///
  /// is equivalent to::
  ///
  ///     Dims(0, 2).IndexArraySlice(MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}),
  ///                                MakeArray<Index>({{7, 8, 9}, {0, 1, 2}}))
  ///
  /// and has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 1}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [0, 9]``
  ///      - ``[0, 1], [0, 2], [2, 5]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"", "", "y"}``
  ///    * - Equivalent input indices
  ///      - ``{1, y, 7}``
  ///      - ``{0, 0, y}``
  ///    * - Equivalent input indices
  ///      - ``{2, y, 8}``
  ///      - ``{0, 1, y}``
  ///    * - Equivalent input indices
  ///      - ``{3, y, 9}``
  ///      - ``{0, 2, y}``
  ///    * - Equivalent input indices
  ///      - ``{6, y, 2}``
  ///      - ``{1, 2, y}``
  ///    * - Equivalent input indices
  ///      - ``{v(a,b,0), y, v(a,b,1)}``
  ///      - ``{a, b, y}``
  ///
  /// where ``y`` is any index in ``[2, 5]``, ``a`` is any index in
  /// ``[0, 1]``, ``b`` is any index in ``[0, 2]``, and::
  ///
  ///     v = MakeArray<Index>({{{1, 7}, {2, 8}, {3, 9}},
  ///                           {{4, 0}, {5, 1}, {6, 2}}})
  ///
  /// \requires `IndexVectorArray` satisfies IsIndexArray and has a non-zero
  ///     rank.
  /// \param index_vector_array The array of index vectors used to index the
  ///     selected dimension.  The new transform may share ownership of the
  ///     supplied index vector array.  The array should not be modified after
  ///     being passed to this function.  The index values contained in the
  ///     array may be bounds-checked lazily.
  /// \param vector_dimension Optional.  The dimension of `index_vector_array`
  ///     that corresponds to the vector of selected dimensions.  May be a
  ///     negative value, as supported by NormalizeDimensionIndex.  If not
  ///     specified, defaults to the last dimension.  The other dimensions of
  ///     `index_vector_array` correspond to new input dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if `index_vector_array.rank()`
  ///     is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if `vector_dimension` is
  ///     invalid.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the dimension
  ///     of `index_vector_array` indicated by `vector_dimension` is not equal
  ///     to the number of selected dimensions.
  /// \error `absl::StatusCode::kOutOfRange` if an out-of-bounds index is
  ///     detected.
  /// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
  ///     when computing the resultant transform.
  template <typename IndexVectorArray>
  IndexVectorArraySliceOpExpr<IndexVectorArray> IndexVectorArraySlice(
      const IndexVectorArray& index_vector_array,
      DimensionIndex vector_dimension = -1) const {
    return {{index_vector_array, vector_dimension}, *this};
  }

  /// Independently slices the selected dimensions using index arrays.
  ///
  /// Each of the ``k = sizeof...(IndexArray)`` ``selected_dimensions``
  /// correspond to an ``index_array``.  Given an ``existing_transform``
  /// with input rank ``m``, the new index transform has an input rank
  /// ``p = m + sum((index_array.rank() - 1)...) - 1``, there is a list
  /// ``identity_dims`` of input dimensions and for each
  /// ``selected_dimensions[i]``, a separate list ``array_dims.[i]`` of
  /// ``index_array.[i].rank()`` consecutive input dimensions such that::
  ///
  ///     flatten(interleave(identity_dims,
  ///                        selected_dimensions, {array_dims...})) == 0:p.
  ///
  /// The new transform maps ``input`` vectors of size ``p`` to::
  ///
  ///     existing_transform(
  ///         interleave(
  ///             input[identity_dims],
  ///             selected_dimensions,
  ///             {index_array(input[array_dims])...}))
  ///
  /// The domain of the new input dimension ``array_dims.[i][j]`` is
  /// ``[0, index_array.[i].size(j))``.  The new dimension selection is
  /// ``flatten({array_dims...})``.
  ///
  /// For example::
  ///
  ///     Dims(2, 0).OuterIndexArraySlice(MakeArray<Index>({{2, 3}, {4, 5}}),
  ///                                     MakeArray<Index>({6, 7}))
  ///
  /// has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dim. selection
  ///      - ``{2, 0}``
  ///      - ``{2, 3, 0}``
  ///    * - Input domain
  ///      - ``[4, 8], [2, 5], [0, 9]``
  ///      - ``[0, 1], [2, 5], [0, 1], [0, 1]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"", "y", "", ""}``
  ///    * - Equivalent inputs
  ///      - ``{6, 3, 3}``
  ///      - ``{0, 3, 0, 1}``
  ///    * - Equivalent inputs
  ///      - ``{7, 3, 4}``
  ///      - ``{1, 3, 1, 0}``
  ///    * - Equivalent inputs
  ///      - ``{xi(a), y, zi(b,c)}``
  ///      - ``{a, y, b, c}``
  ///
  /// where ``y`` is any index in ``[2, 5]``, ``a`` is any index in
  /// ``[0, 1]``, ``b`` is any index in ``[0, 1]``, ``c`` is any
  /// index in ``[0, 1]``, ``xi = MakeArray<Index>({6, 7}`` and
  /// ``zi = MakeArray<Index>({{2, 3}, {4, 5}})``.
  ///
  /// \requires `(IsIndexArray<IndexArray> && ...)`
  /// \requires `sizeof...(IndexArray)` must be compatible with the static rank
  ///     of the dimension selection.
  /// \param index_arrays The index arrays used to index into each selected
  ///     dimension.  The new transform may share ownership of the supplied
  ///     index arrays.  The index arrays should not be modified after being
  ///     passed to this function.  The index values contained in the index
  ///     arrays may be bounds-checked lazily.  May also be specified as a
  ///     `span`.
  /// \error `absl::StatusCode::kInvalidArgument` if `sizeof...(IndexArray)` is
  ///     not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kOutOfRange` if an out-of-bounds index is
  ///     detected. \error `absl::StatusCode::kInvalidArgument` if integer
  ///     overflow occurs when computing the resultant transform.
  template <typename... IndexArray>
  IndexArrayOuterSliceOpExpr<IndexArray...> OuterIndexArraySlice(
      const IndexArray&... index_arrays) const {
    return {{index_arrays...}, *this};
  }
  DynamicIndexArrayOuterSliceOpExpr OuterIndexArraySlice(
      span<const SharedArrayView<const Index>> index_arrays) const {
    return {{index_arrays}, *this};
  }

  /// Sets (or changes) the labels of the selected dimensions.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static rank of the `labels` vector.
  ///
  /// For example: `Dims(0, 2).Label({"a", "b"})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"a", "y", "b"}``
  ///
  /// \requires `Labels` is `span`-compatible with a `span::value_type` of
  ///     `std::string`, `std::string_view`, or `const char *`, and a static
  ///     extent compatible with the static rank of the dimension selection.
  /// \param labels The new labels for each of the selected dimensions.  May be
  ///     a braced list, e.g. `Label({"a", "b"})`.  May also be specified as an
  ///     argument pack, e.g. `Label("a", "b", "c")`.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `labels`
  ///     vector is equal to the number of selected dimensions.
  template <typename Labels>
  LabelSpanOpExpr<Labels> Label(const Labels& labels) const {
    return {{labels}, *this};
  }
  template <typename... L>
  LabelPackOpExpr<L...> Label(const L&... labels) const {
    return {{{{labels...}}}, *this};
  }

  // Overload that permits the labels to specified as a braced list.
  template <DimensionIndex Rank>
  LabelOpExpr<span<const std::string_view, Rank>, Rank> Label(
      const std::string_view (&labels)[Rank]) const {
    return {{labels}, *this};
  }

  /// Transposes the input dimensions such that the selected dimensions are
  /// consecutive starting or ending at the specified `target` position.
  ///
  /// For example, `Dims(2, 0).MoveTo(1)` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{1, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[2, 5], [3, 4], [1, 3]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"y", "z", "x"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 4}``
  ///      - ``{3, 4, 2}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{y, z, x}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \param target Must be in the range
  ///     ``[-input_rank + selection_rank - 1, input_rank - selection_rank]``.
  ///     If ``target >= 0``, ``target`` is the new index of the first selected
  ///     dimension.  If ``target < 0``, ``target + input_rank`` is the new
  ///     index of the last selected dimension.
  /// \error `absl::StatusCode::kInvalidArgument` if `target` is outside the
  ///     valid range.
  MoveToOpExpr MoveTo(DimensionIndex target) const { return {{target}, *this}; }

  /// Equivalent to `MoveTo(0)`.
  MoveToOpExpr MoveToFront() const { return {{0}, *this}; }

  /// Equivalent to `MoveTo(-1)`.
  MoveToOpExpr MoveToBack() const { return {{-1}, *this}; }

  /// Extracts the diagonal of the selected dimensions.
  ///
  /// The selected dimensions are removed from the new index space, and a new
  /// dimension corresponding to the diagonal is added as the first dimension,
  /// with an input domain equal to the intersection of the input domains of the
  /// selected dimensions.  The new dimension selection is equal to ``{0}``,
  /// corresponding to the newly added diagonal dimension.
  ///
  /// The lower and upper bounds of the new diagonal dimension are implicit if,
  /// and only if, the lower or upper bounds, respectively, of every selected
  /// dimension are implicit.
  ///
  /// For example: `Dims(0, 2).Diagonal()` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0}``
  ///    * - Input domain
  ///      - ``[1, 5], [2, 5], [3, 7]``
  ///      - ``[3, 5], [2, 5]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"", "y"}``
  ///    * - Equivalent input indices
  ///      - ``{4, 3, 4}``
  ///      - ``{4, 3}``
  ///    * - Equivalent input indices
  ///      - ``{d, y, d}``
  ///      - ``{d, y}``
  ///
  /// where ``d`` is any index in ``[3, 5]`` and ``y`` is any index in
  /// ``[2, 5]``.  Note that the domain of the new dimension corresponding to
  /// the diagonal is the intersection of the domains of the ``"x"`` and
  /// ``"z"`` dimensions.
  ///
  /// \remark `Diagonal()` with zero selected dimensions adds a new dummy
  ///     dimension as the first dimension.
  /// \remark `Diagonal()` with a single selected dimension is equivalent to
  ///     `MoveToFront().Label("")`.
  DiagonalOpExpr Diagonal() const { return {{}, *this}; }

  /// Adds new dummy input dimensions that have no effect on the output indices.
  ///
  /// The added dimensions have a domain of ``(-kInfIndex, kInfIndex)``,
  /// which can be reduced by chaining a call to `ClosedInterval`,
  /// `SizedInterval`, or `HalfOpenInterval`.  The lower and upper bounds of the
  /// added dimensions are implicit.  The labels of the added dimensions (if
  /// applicable) are empty.
  ///
  /// The prior dimension selection (e.g. specified by a call to Dims on which
  /// the call to AddNew is chained) specifies the indices of the new
  /// dimensions, in the range
  /// ``(-input_rank - selection_rank, input_rank + selection_rank)``.  This
  /// is unlike every other operation, for which the prior dimension selection
  /// specifies existing dimensions.  The new dimension selection is equal to
  /// the prior dimension selection.
  ///
  /// Because the prior dimension selection specifies new rather than existing
  /// dimensions, this operation cannot be chained after any other operation
  /// (but other operations may be changed after it).
  ///
  /// For example, `Dims(0, -1).AddNew()` (equivalent to `Dims(0, 2).AddNew()`)
  /// has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 5]``
  ///      - ``[-inf*, +inf*], [1, 5], [-inf*, +inf*]``
  ///    * - Labels
  ///      - ``{"x"}``
  ///      - ``{"", "x", ""}``
  ///    * - Equivalent input indices
  ///      - ``{2}``
  ///      - ``{1, 2, 8}``
  ///    * - Equivalent input indices
  ///      - ``{x}``
  ///      - ``{a, x, b}``
  ///
  /// where ``x`` is any index in ``[1, 5]``, ``a`` is any index, and
  /// ``b`` is any index.
  ///
  /// \requires There is no prior operation in the sequence.
  /// \requires The prior selected dimensions must have been specified by index,
  ///     rather than by label, and must not have been specified using the
  ///     `AllDims()` selection.
  template <int&... ExplicitArgumentBarrier,
            bool SfinaeIsFirst = sizeof...(Op) == 1>
  std::enable_if_t<SfinaeIsFirst, AddNewOpExpr> AddNew() const {
    return {{}, *this};
  }

  /// Transposes the input dimensions such that the selected dimensions are
  /// consecutive.
  ///
  /// This is equivalent to `MoveToFront()` and `MoveToBack()`, but requires
  /// that all dimensions are selected.  The new dimension selection is
  /// ``{0, ..., input_rank-1}``.
  ///
  /// For example, `Dims(2, 0, 1).Transpose()` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{2, 0, 1}``
  ///      - ``{0, 1, 2}``
  ///    * - Input domain
  ///      - ``[1*, 3], [2, 5*], [3, 4]``
  ///      - ``[3, 4], [1*, 3], [2, 5*]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"z", "x", "y"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 4}``
  ///      - ``{4, 2, 3}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{z, x, y}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \requires The static rank of the dimension selection must be compatible
  ///     with the static input rank.
  /// \error `absl::StatusCode::kInvalidArgument` if the rank of the dimension
  ///     selection is not equal to the input rank.
  /// \id consecutive
  TransposeOpExpr Transpose() const { return {{}, *this}; }

  /// Transposes the input dimensions such that the selected dimensions have the
  /// specified indices.  Dimensions not in the selection retain their relative
  /// order and fill in the dimension indices not in `target_dimensions`.
  ///
  /// The new dimension selection is equal to `target_dimensions` after
  /// normalization.
  ///
  /// For example, `Dims(2, 0).Transpose({1,2})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{2, 0}``
  ///      - ``{1, 2}``
  ///    * - Input domain
  ///      - ``[1*, 3], [2, 5*], [3, 4]``
  ///      - ``[2, 5*], [3, 4], [1*, 3]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"y", "z", "x"}``
  ///    * - Equivalent input indices
  ///      - ``{2, 3, 4}``
  ///      - ``{3, 4, 2}``
  ///    * - Equivalent input indices
  ///      - ``{x, y, z}``
  ///      - ``{y, z, x}``
  ///
  /// where ``x`` is any index in ``[1, 3]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[3, 4]``.
  ///
  /// \requires `TargetDimensions` is `span`-compatible with a
  ///     `span::value_type` of `DimensionIndex` and a static extent compatible
  ///     with the static rank of the dimension selection.
  /// \param target_dimensions The new dimension indices corresponding to each
  ///     selected dimension.  May be a braced list, e.g. `Transpose({1, 2})`.
  ///     A negative value of ``-n`` is equivalent to ``input_rank - n``,
  ///     where ``input_rank`` is the input rank of the transform to which
  ///     this operation is applied.
  /// \error `absl::StatusCode::kInvalidArgument` if the rank of the dimension
  ///     selection is not equal to the length of `target_dimensions`, or if the
  ///     indices in `target_dimensions` are not unique or outside the valid
  ///     range.
  /// \id target_dimensions
  template <typename TargetDimensions>
  TransposeToOpExpr<TargetDimensions> Transpose(
      const TargetDimensions& target_dimensions) const {
    return {{span(target_dimensions)}, *this};
  }

  // Overload that permits the target dimensions to be specified as a braced
  // list.
  template <DimensionIndex Rank>
  TransposeToOpExpr<span<const DimensionIndex, Rank>> Transpose(
      const DimensionIndex (&target_dimensions)[Rank]) const {
    return {{span(target_dimensions)}, *this};
  }

  /// Marks the lower and/or upper bounds of the selected dimensions as
  /// explicit.
  ///
  /// The new dimension selection is the same as the prior dimension selection.
  ///
  /// For example: `Dims(0, 2).MarkBoundsExplicit()` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3*], [2*, 5], [3*, 4]``
  ///      - ``[1, 3], [2*, 5], [3, 4]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///
  /// \param lower If `true` (the default), mark the lower bounds as explicit.
  /// \param upper If `true` (the default), mark the upper bounds as explicit.
  ChangeImplicitStateOpExpr MarkBoundsExplicit(bool lower = true,
                                               bool upper = true) const {
    return {{/*.implicit=*/false, lower, upper}, *this};
  }

  /// Marks the lower and/or upper bounds of the selected dimensions as
  /// implicit.
  ///
  /// This operation is potentially unsafe because it may be used to bypass
  /// normal bounds checking.
  ///
  /// The new dimension selection is the same as the prior dimension selection.
  ///
  /// For example: `Dims(0, 2).UnsafeMarkBoundsImplicit()` has the following
  /// effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[1, 3], [2, 5], [3, 4]``
  ///      - ``[1*, 3*], [2, 5], [3*, 4*]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///
  /// \param lower If `true` (the default), mark the lower bounds as implicit.
  /// \param upper If `true` (the default), mark the upper bounds as implicit.
  ChangeImplicitStateOpExpr UnsafeMarkBoundsImplicit(bool lower = true,
                                                     bool upper = true) const {
    return {{/*.implicit=*/true, lower, upper}, *this};
  }

  /// Strides the domains of the selected input dimensions by the specified
  /// `strides` vector.
  ///
  /// For each selected dimension ``i``, the new domain is the set of indices
  /// ``x`` such that ``x * strides[i]`` is contained in the original
  /// domain.
  ///
  /// This has the same effect as `SizedInterval(kImplicit, kImplicit, strides)`
  /// except that the domain may be translated by 1 and does not require a
  /// bounded start index.
  ///
  /// The new dimension selection is the same as the prior dimension selection,
  /// with a static rank equal to the merged static rank of the prior dimension
  /// selection and the static extent of the `strides` vector.
  ///
  /// For example: `Dims(0, 2).Stride({-2, 3})` has the following effects:
  ///
  /// .. list-table::
  ///    :header-rows: 1
  ///
  ///    * -
  ///      - Before
  ///      - After
  ///    * - Dimension selection
  ///      - ``{0, 2}``
  ///      - ``{0, 2}``
  ///    * - Input domain
  ///      - ``[0, 6], [2, 5], [1, 8]``
  ///      - ``[-3, 0], [2, 5], [1, 2]``
  ///    * - Labels
  ///      - ``{"x", "y", "z"}``
  ///      - ``{"x", "y", "z"}``
  ///    * - Equivalent input indices
  ///      - ``{4, 3, 3}``
  ///      - ``{-2, 3, 1}``
  ///    * - Equivalent input indices
  ///      - ``{-2 * x, y, 3 * z}``
  ///      - ``{x, y, z}``
  ///
  /// where ``x`` is any index in ``[-3, 0]``, ``y`` is any index in
  /// ``[2, 5]``, and ``z`` is any index in ``[1, 2]``.
  ///
  /// \requires `Strides` satisfies the IsIndexVectorOrScalar concept with a
  ///     static extent compatible with the static rank of the dimension
  ///     selection.
  /// \param strides Index vector specifying the stride for each selected
  ///     dimension.  May be a braced list, e.g. `Stride({1, 2, 3})`.  May also
  ///     be a scalar, e.g. `Stride(5)`, in which case the same stride is used
  ///     for all selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if the extent of the `strides`
  ///     vector is not equal to the number of selected dimensions.
  /// \error `absl::StatusCode::kInvalidArgument` if a stride value is `0`.
  template <typename Strides>
  StrideOpExpr<Strides> Stride(const Strides& strides) const {
    return {{strides}, *this};
  }

  // Overload that permits the strides vector to be specified as a braced list.
  template <DimensionIndex Rank>
  StrideOpExpr<const Index (&)[Rank]> Stride(
      const Index (&strides)[Rank]) const {
    return {{span(strides)}, *this};
  }

  // TODO(jbms): Add Squeeze operation

  // Type alias for the new transform type that results from applying this
  // DimExpression to an index transform with the specified `InputRank`,
  // `OutputRank`.
  template <DimensionIndex InputRank, DimensionIndex OutputRank>
  using NewTransformType =
      DimExpressionHelper::NewTransformType<InputRank, OutputRank, Op...>;

  // Type alias for the new domain type that results from applying this
  // DimExpression to an index domain with the specified `Rank`.
  template <DimensionIndex Rank>
  using NewDomainType = DimExpressionHelper::NewDomainType<Rank, Op...>;

  /// Applies this DimExpression to the specified index transform.
  ///
  /// \requires This DimExpression contains at least one operation.
  /// \param transform The index transform to which this DimExpression is
  ///     applied.
  /// \param selection_output[out] Optional.  If specified, filled with the
  ///     indices of the new dimension selection after applying this
  ///     DimExpression.
  /// \returns The new index transform, or any error caused by the initial
  ///     dimension selection or one of the chained operations.
  /// \id transform
  template <DimensionIndex InputRank, DimensionIndex OutputRank,
            ContainerKind CKind>
  Result<NewTransformType<InputRank, OutputRank>> operator()(
      IndexTransform<InputRank, OutputRank, CKind> transform,
      DimensionIndexBuffer* selection_output =
          &internal::GetLValue(DimensionIndexBuffer())) const {
    // NONITPICK: internal
    // NONITPICK: internal::GetLValue
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto result,
        DimExpressionHelper::Apply(*this, std::move(transform),
                                   selection_output, /*domain_only=*/false));
    return NewTransformType<InputRank, OutputRank>(unchecked,
                                                   std::move(result));
  }

  /// Applies this DimExpression to the specified index domain.
  ///
  /// \requires This DimExpression contains at least one operation.
  /// \param domain The index domain to which this DimExpression is applied.
  /// \param selection_output[out] Optional.  If specified, filled with the
  ///     indices of the new dimension selection after applying this
  ///     DimExpression.
  /// \returns The new index domain, or any error caused by the initial
  ///     dimension selection or one of the chained operations.
  /// \id domain
  template <DimensionIndex Rank, ContainerKind CKind>
  Result<NewDomainType<Rank>> operator()(
      IndexDomain<Rank, CKind> domain,
      DimensionIndexBuffer* selection_output =
          &internal::GetLValue(DimensionIndexBuffer())) const {
    // NONITPICK: internal
    // NONITPICK: internal::GetLValue
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto result,
        DimExpressionHelper::Apply(*this, Access::transform(std::move(domain)),
                                   selection_output, /*domain_only=*/true));
    return Access::Make<NewDomainType<Rank>>(
        Access::rep_ptr(std::move(result)));
  }

  /// Applies this DimExpression to an object with an associated index space
  /// that supports `ApplyIndexTransform`.
  ///
  /// \id transformable
  template <typename Transformable>
  internal_index_space::EnableIfApplyIndexTransformResult<
      !IsIndexTransform<internal::remove_cvref_t<Transformable>>,
      const DimExpression&, Transformable>
  operator()(Transformable&& transformable) const {
    return ApplyIndexTransform(*this,
                               std::forward<Transformable>(transformable));
  }

  template <DimensionIndex InputRank, DimensionIndex OutputRank,
            ContainerKind CKind>
  friend Result<NewTransformType<InputRank, OutputRank>> ApplyIndexTransform(
      const DimExpression& expr,
      IndexTransform<InputRank, OutputRank, CKind> transform) {
    return expr(std::move(transform));
  }

  /// Resolves a dimension selection to dimension indices.
  ///
  /// For example::
  ///
  ///     auto transform = IdentityTransform({"x", "y", "z"});
  ///     DimensionIndexBuffer buffer;
  ///     TENSORSTORE_EXPECT_OK(Dims("x", "z").Resolve(transform.domain(),
  ///                                                  &buffer));
  ///     EXPECT_THAT(buffer, ::testing::ElementsAre(0, 2));
  ///
  /// \param domain The domain for which to resolve the dimension selection.
  /// \param selection_output[out] Non-null pointer to buffer that will be
  ///     filled with dimension indices.
  /// \requires There is no prior operation in the sequence.
  /// \error `absl::StatusCode::kInvalidArgument` if this dimension selection is
  ///     not compatible with `domain`.
  template <DimensionIndex Rank, ContainerKind CKind>
  std::enable_if_t<internal_index_space::DimExpressionHelper::
                       CanResolveDimensions<Rank, Op...>,
                   absl::Status>
  Resolve(const IndexDomain<Rank, CKind>& domain,
          DimensionIndexBuffer* selection_output) const {
    return last_op_.GetDimensions(
        internal_index_space::TransformAccess::transform(domain),
        selection_output);
  }

  // Constructs a DimExpression from a new op and the parent DimExpression (if
  // any) to which it should be chained.  This is for internal use only.
  DimExpression(LastOp last_op = {}, const Parent& parent = {})
      : last_op_(std::move(last_op)), parent_(parent) {}

 private:
  template <typename...>
  friend class DimExpression;

  friend class internal_index_space::DimExpressionHelper;

  LastOp last_op_;
  Parent parent_;
};

// Specialization of DimExpression for an empty parameter list.  This is an
// implementation detail, and is not a part of the public API.
template <>
class DimExpression<> {
  template <typename...>
  friend class DimExpression;
  friend class internal_index_space::DimExpressionHelper;

 public:
  DimExpression() = default;
};

/// Starts a `DimExpression` with the specified dimensions selected (and no
/// operations).
///
/// The static rank of the dimension selection of the resultant DimExpression is
/// equal to the static extent of `dimensions`.
///
/// Dimensions may specified by an index in the range
/// ``(-input_rank, input_rank)``.  Negative dimension indices are normalized
/// by adding ``input_rank``.  Dimensions may also be specified by a
/// non-empty string label.
///
/// \requires For the single-parameter overload, `Dimensions` is
///     `span`-compatible with a `span::value_type` equal to `DimensionIndex`,
///     `DimensionIdentifier`, or `DynamicDimSpec`.
/// \requires For the variadic overload, each `DimensionId` type must be
///     convertible without narrowing to `DimensionIndex`,
///     `DimensionIdentifier`, or `DynamicDimSpec`.
/// \param dimensions The dimension identifiers.  May also be a braced list,
///     e.g. `Dims({1, 2})`, or a pack, e.g. `Dims(1, "x")`.
/// \error `absl::StatusCode::kInvalidArgument` if a dimension index is outside
///     the valid range, or a specified label is empty or does not equal a
///     dimension label.
/// \relates DimExpression
template <typename Dimensions>
inline DimExpression<
    internal_index_space::DimensionListFromSpanType<Dimensions>>
Dims(const Dimensions& dimensions) {
  return {{span(dimensions)}};
}
template <typename... DimensionId>
inline DimExpression<
    internal_index_space::DimensionsFromPackType<DimensionId...>>
Dims(const DimensionId&... dimensions) {
  return {{{{dimensions...}}}};
}

// Overload that permits dimensions to be specified by index using a braced
// list, e.g. `Dims({1, 2})`.
template <DimensionIndex Rank>
inline DimExpression<internal_index_space::DimensionListFromSpanType<
    span<const DimensionIndex, Rank>>>
Dims(const DimensionIndex (&dimensions)[Rank]) {
  return {{span(dimensions)}};
}

// Overload that permits dimensions to be specified by index or label using a
// braced list, e.g. `Dims({"x", 3})`.
template <DimensionIndex Rank>
inline DimExpression<internal_index_space::DimensionListFromSpanType<
    span<const DimensionIdentifier, Rank>>>
Dims(const DimensionIdentifier (&dimensions)[Rank]) {
  return {{span(dimensions)}};
}

/// Starts a `DimExpression` with a range of dimensions.
///
/// See `DimRangeSpec` for more detailed documentation.
///
/// - `DimRange(3)` specifies all dimensions greater than 3.
///
/// - `DimRange(3, 6)` specifies dimensions  {3, 4, 5}.
///
/// - `DimRange(3, std::nullopt, 2)` specifies odd dimensions greater than 3.
///
/// - `DimRange(-3)` specifies the last 3 dimensions, and can be used
///   to add 3 new trailing dimensions.
///
/// - `DimRange(1, -2)` specifies dimensions 1 up to but not including
///   the second from the last.  It cannot be used to infer the final rank when
///   adding new dimensions.
///
/// Since `DimRange` resolves to a variable number of dimensions, the resulting
/// `DimExpression` always has a compile-time rank of `dynamic_rank`.
///
/// \relates DimExpression
inline DimExpression<
    internal_index_space::DimensionList<std::array<DynamicDimSpec, 1>>>
DimRange(DimensionIndex inclusive_start,
         std::optional<DimensionIndex> exclusive_stop = std::nullopt,
         const DimensionIndex step = 1) {
  return {{DimRangeSpec{inclusive_start, exclusive_stop, step}}};
}

/// Starts a `DimExpression` with a variable number of dimensions.
///
/// This permits dynamic dimension specifications to be specified using
/// a braced list, e.g. `DynamicDims({DimRangeSpec(1, 5, 2), 5, 7, "x"})`.
///
/// Since a `DynamicDimSpec` containing a `DimRangeSpec` resolves to a variable
/// number of dimensions, the resultant dimension selection always has a
/// compile-time rank of `dynamic_rank`.
///
/// \relates DimExpression
template <DimensionIndex Rank>
inline DimExpression<
    internal_index_space::DimensionList<span<const DynamicDimSpec, Rank>>>
DynamicDims(const DynamicDimSpec (&dimensions)[Rank]) {
  return {{span(dimensions)}};
}

/// Starts a `DimExpression` with all dimensions selected (and no operations).
///
/// \relates DimExpression
inline DimExpression<internal_index_space::AllDims> AllDims() { return {}; }

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_DIM_EXPRESSION_H_
