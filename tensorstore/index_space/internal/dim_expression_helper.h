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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_HELPER_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_HELPER_H_

/// \file
/// Implementation details for DimExpression.

#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

template <typename... Op>
class DimExpression;

namespace internal_index_space {

/// Evaluates to `true` if, and only if, `IndexVector...` all satisfy
/// `IsIndexVectorOrScalar<IndexVector>::value` and have a static extent
/// compatible with `StaticSelectionRank`.  The `SFINAE` argument must be
/// `void`.
///
/// This is defined separately rather than inlined below to work around a bug in
/// MSVC 2019 regarding variadic template arguments:
/// https://developercommunity.visualstudio.com/content/problem/915028/error-with-variadic-template-expansion-in-function.html
template <DimensionIndex StaticSelectionRank, typename SFINAE,
          typename... IndexVector>
constexpr bool IndexVectorsCompatible = false;

template <DimensionIndex StaticSelectionRank, typename... IndexVector>
constexpr bool IndexVectorsCompatible<
    StaticSelectionRank,
    std::enable_if_t<(IsIndexVectorOrScalar<IndexVector>::value && ...)>,
    IndexVector...> =
    RankConstraint::EqualOrUnspecified(
        {StaticSelectionRank, IsIndexVectorOrScalar<IndexVector>::extent...});

template <typename Last, typename... Prior>
struct DimExpressionTraits {
  using LastOp = Last;
  using Parent = DimExpression<Prior...>;
};

/// Helper friend class used by DimExpression to apply operations to an
/// IndexTransform.
class DimExpressionHelper {
 public:
  /// Returns the static rank of the resultant dimension selection.
  ///
  /// Depending on how the operation sequence and how the initial dimension
  /// selection was specified, the rank of the dimension selection may or may
  /// not depend on the `input_rank` of the transform.
  ///
  /// \tparam Op The operation sequence.
  /// \param input_rank static input rank of the index transform to which the
  ///     DimExpression will be applied, or `dynamic_rank` if it is unknown.
  /// \scheck `input_rank` is compatible with the operation sequence.
  /// \returns The static rank of the dimension selection after applying the
  ///     sequence of operations to an index transform with the specified
  ///     `input_rank`.
  template <typename... Op>
  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    DimensionIndex selection_rank = input_rank;
#ifndef _MSC_VER
    // The folding expression below evaluates terms in reverse order due to
    // right-to-left evaluation order of `operator=`.
    [[maybe_unused]] int unused = 0;
    (unused = ... =
         (selection_rank = Op::GetStaticSelectionRank(selection_rank), 0));
    return selection_rank;
#else
    // Use recursive implementation to workaround
    // https://developercommunity.visualstudio.com/t/operator-incorrectly-evaluated-left-to/10307305
    return GetStaticSelectionRankRecursive<Op...>(selection_rank);
#endif
  }

#ifdef _MSC_VER
  template <typename LastOp, typename... PriorOp>
  constexpr static DimensionIndex GetStaticSelectionRankRecursive(
      DimensionIndex input_rank) {
    if constexpr (sizeof...(PriorOp) == 0) {
      return LastOp::GetStaticSelectionRank(input_rank);
    } else {
      return LastOp::GetStaticSelectionRank(
          GetStaticSelectionRankRecursive<PriorOp...>(input_rank));
    }
  }
#endif

  /// Returns the static rank of the resultant index transform.
  ///
  /// \tparam Op... sequence of operations to apply.
  /// \param input_rank static input rank of the index transform to which the
  ///     DimExpression will be applied.
  template <typename... Op>
  constexpr static DimensionIndex GetNewStaticInputRank(
      DimensionIndex input_rank) {
    DimensionIndex selection_rank = input_rank;
#ifndef _MSC_VER
    // The folding expression below evaluates terms in reverse order due to
    // right-to-left evaluation order of `operator=`.
    [[maybe_unused]] int unused = 0;
    (unused = ... =
         (input_rank = Op::GetNewStaticInputRank(input_rank, selection_rank),
          selection_rank = Op::GetStaticSelectionRank(selection_rank), 0));
    return input_rank;
#else
    return GetNewStaticInputRankRecursive<Op...>(input_rank, selection_rank);
#endif
  }

#ifdef _MSC_VER
  template <typename LastOp, typename... PriorOp>
  constexpr static DimensionIndex GetNewStaticInputRankRecursive(
      DimensionIndex input_rank, DimensionIndex& selection_rank) {
    if constexpr (sizeof...(PriorOp) == 0) {
      selection_rank = LastOp::GetStaticSelectionRank(input_rank);
      return input_rank;
    } else {
      input_rank = GetNewStaticInputRankRecursive<PriorOp...>(input_rank,
                                                              selection_rank);
      input_rank = LastOp::GetNewStaticInputRank(input_rank, selection_rank);
      selection_rank = LastOp::GetStaticSelectionRank(selection_rank);
      return input_rank;
    }
  }
#endif

  /// Type of index transform obtained by applying a DimExpression.
  ///
  /// \tparam InputRank static input rank of the index transform to which the
  ///     DimExpression will be applied.
  /// \tparam OutputRank static output rank of the index transform to which the
  ///     DimExpression will be applied.
  /// \tparam Op... Sequence of operations that will be applied.
  /// \schecks `InputRank` is compatible with `DimExpression<Op...>`.
  template <DimensionIndex InputRank, DimensionIndex OutputRank, typename... Op>
  using NewTransformType =
      IndexTransform<GetNewStaticInputRank<Op...>(InputRank), OutputRank>;

  /// Type of index domain obtained by applying a DimExpression.
  ///
  /// \tparam Rank static rank of the index domain to which the DimExpression
  ///     will be applied.
  /// \tparam Op... Sequence of operations that will be applied.
  /// \schecks `Rank` is compatible with `DimExpression<Op...>`.
  template <DimensionIndex Rank, typename... Op>
  using NewDomainType = IndexDomain<GetNewStaticInputRank<Op...>(Rank)>;

  /// Applies a DimExpression to an index transform.
  template <typename LastOp, typename PriorOp0, typename PriorOp1,
            typename... PriorOp>
  static Result<IndexTransform<>> Apply(
      const DimExpression<LastOp, PriorOp0, PriorOp1, PriorOp...>& expr,
      IndexTransform<> transform, DimensionIndexBuffer* dimensions,
      bool domain_only) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform,
        Apply(expr.parent_, std::move(transform), dimensions, domain_only));
    return expr.last_op_.Apply(std::move(transform), dimensions, domain_only);
  }

  /// Overload for the base case of a single operation.
  template <typename DimensionSelection, typename Op>
  static Result<IndexTransform<>> Apply(
      const DimExpression<Op, DimensionSelection>& expr,
      IndexTransform<> transform, DimensionIndexBuffer* dimensions,
      bool domain_only) {
    TENSORSTORE_RETURN_IF_ERROR(GetDimensions<Op::selected_dimensions_are_new>(
        expr.parent_.last_op_, transform, dimensions));
    return expr.last_op_.Apply(std::move(transform), dimensions, domain_only);
  }

  /// Sets `*output` to the selection of existing dimensions.
  ///
  /// This is used to obtain the initial dimension selection for most
  /// operations.
  template <bool SelectedDimensionsAreNew, typename DimensionSelection>
  static std::enable_if_t<!SelectedDimensionsAreNew, absl::Status>
  GetDimensions(const DimensionSelection& selection,
                IndexTransformView<> transform, DimensionIndexBuffer* output) {
    return selection.GetDimensions(transform, output);
  }

  /// Sets `*output` to the selection of new (not existing) dimensions.
  ///
  /// This is used to obtain the initial dimension selection for the AddNew
  /// operation.
  template <bool SelectedDimensionsAreNew, typename DimensionSelection>
  static std::enable_if_t<SelectedDimensionsAreNew, absl::Status> GetDimensions(
      const DimensionSelection& selection, IndexTransformView<> transform,
      DimensionIndexBuffer* output) {
    return selection.GetNewDimensions(transform.input_rank(), output);
  }

  template <template <typename...> class OpTemplate,
            DimensionIndex StaticSelectionRank, typename... IndexVector>
  using IndexVectorOp = std::enable_if_t<
      IndexVectorsCompatible<StaticSelectionRank, void, IndexVector...>,
      OpTemplate<
          typename IsIndexVectorOrScalar<IndexVector>::normalized_type...>>;

  template <DimensionIndex InputRank, typename... Op>
  static constexpr inline bool CanResolveDimensions =
      (sizeof...(Op) == 1 && GetStaticSelectionRank<Op...>(InputRank) >= -1);
};

// Used to implement `EnableIfApplyIndexTransformResult` below.
template <bool Condition>
struct ConditionalApplyIndexTransformResult {
  template <typename Expr, typename Transformable>
  using type = decltype(ApplyIndexTransform(std::declval<Expr>(),
                                            std::declval<Transformable>()));
};
template <>
struct ConditionalApplyIndexTransformResult<false> {};

/// Equivalent to:
///
///     std::enable_if_t<
///         Condition,
///         decltype(ApplyIndexTransform(std::declval<Expr>(),
///                                      std::declval<Transformable>()))>
///
/// except that the `decltype` is only evaluated if `Condition` is `true` (this
/// avoids the potential for SFINAE loops).
template <bool Condition, typename Expr, typename Transformable>
using EnableIfApplyIndexTransformResult =
    typename ConditionalApplyIndexTransformResult<Condition>::template type<
        Expr, Transformable>;

}  // namespace internal_index_space

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_HELPER_H_
