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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_TESTUTIL_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_TESTUTIL_H_

#include <string>
#include <system_error>  // NOLINT
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal_index_space {

using EquivalentIndices =
    std::vector<std::pair<std::vector<Index>, std::vector<Index>>>;

template <typename A, typename B>
inline void CheckSameTypes() {
  static_assert(std::is_same_v<A, B>);
}

template <typename Transform>
Transform MakeNewTransformCopy(const Transform& t) {
  return TransformAccess::Make<Transform>(
      MutableRep(TransformAccess::rep_ptr<container>(t)));
}

/// Tests applying a `DimExpression` to a transform in place.
///
/// In all cases, checks that the result is equal to the expected value.  If
/// `can_operate_in_place==true`, additionally checks that the operation was
/// actually in place (for some operations, the implementation must make a
/// copy).
///
/// \param original_transform Initial transform
/// \param expression `DimExpression` object to apply
/// \param expected_new_dimension_selection Expected dimension selection after
///     applying transform
/// \param expected_new_transform Expected result of applying `expression` to
///     `original_transform`.
/// \param can_operate_in_place If `true`, `expression` is expected to operate
///     in place when applied to a transform with a reference count of 1.
template <typename OriginalTransform, typename Expression,
          typename ExpectedResult>
void TestDimExpressionInplace(
    const OriginalTransform& original_transform, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedResult& expected_new_transform, bool can_operate_in_place) {
  auto original_copy = MakeNewTransformCopy(original_transform);
  tensorstore::DimensionIndexBuffer dimensions;
  TransformRep* rep = TransformAccess::rep(original_copy);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result_transform_inplace,
      expression(std::move(original_copy), &dimensions));
  const bool operated_in_place =
      (rep == TransformAccess::rep(result_transform_inplace));
  EXPECT_EQ(can_operate_in_place, operated_in_place);
  EXPECT_EQ(expected_new_transform, result_transform_inplace);
  EXPECT_THAT(dimensions,
              testing::ElementsAreArray(expected_new_dimension_selection));
}

/// Same as above, but for IndexDomain rather than IndexTransform.
template <typename OriginalDomain, typename Expression, typename ExpectedResult>
void TestDimExpressionInplaceDomainOnly(
    const OriginalDomain& original_domain, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedResult& expected_new_domain, bool can_operate_in_place) {
  auto original_copy =
      MakeNewTransformCopy<IndexDomain<OriginalDomain::static_rank>>(
          original_domain);
  tensorstore::DimensionIndexBuffer dimensions;
  TransformRep* rep = TransformAccess::rep(original_copy);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result_domain_inplace,
      expression(std::move(original_copy), &dimensions));
  const bool operated_in_place =
      (rep == TransformAccess::rep(result_domain_inplace));
  EXPECT_EQ(can_operate_in_place, operated_in_place);
  EXPECT_EQ(expected_new_domain, result_domain_inplace);
  EXPECT_THAT(dimensions,
              testing::ElementsAreArray(expected_new_dimension_selection));
  if (result_domain_inplace != original_domain) {
    TransformRep* new_rep = TransformAccess::rep(result_domain_inplace);
    EXPECT_EQ(0, new_rep->output_rank);
  }
}

/// Tests applying a `DimExpression` to a transform out of place.
///
/// \param original_transform Initial transform
/// \param expression `DimExpression` object to apply
/// \param expected_new_dimension_selection Expected dimension selection after
///     applying transform
/// \param expected_new_transform Expected result of applying `expression` to
///     `original_transform`.
template <typename OriginalTransform, typename Expression,
          typename ExpectedResult>
void TestDimExpressionOutOfPlace(
    const OriginalTransform& original_transform, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedResult& expected_new_transform) {
  auto original_copy = MakeNewTransformCopy(original_transform);
  tensorstore::DimensionIndexBuffer dimensions;
  auto result_transform = expression(original_transform, &dimensions).value();
  CheckSameTypes<decltype(result_transform), ExpectedResult>();
  if (original_transform != expected_new_transform) {
    EXPECT_NE(TransformAccess::rep(result_transform),
              TransformAccess::rep(original_transform));
  }
  EXPECT_EQ(expected_new_transform, result_transform);
  EXPECT_EQ(original_copy, original_transform);
  EXPECT_THAT(dimensions,
              testing::ElementsAreArray(expected_new_dimension_selection));
}

template <typename OriginalDomain, typename Expression, typename ExpectedResult>
void TestDimExpressionOutOfPlaceDomainOnly(
    const OriginalDomain& original_domain, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedResult& expected_new_domain) {
  auto original_copy =
      MakeNewTransformCopy<IndexDomain<OriginalDomain::static_rank>>(
          original_domain);
  tensorstore::DimensionIndexBuffer dimensions;
  auto result_domain = expression(original_domain, &dimensions).value();
  CheckSameTypes<decltype(result_domain),
                 IndexDomain<ExpectedResult::static_rank>>();
  if (original_domain != expected_new_domain) {
    EXPECT_NE(TransformAccess::rep(result_domain),
              TransformAccess::rep(original_domain));
  }
  EXPECT_EQ(expected_new_domain, result_domain);
  EXPECT_EQ(original_copy, original_domain);
  EXPECT_THAT(dimensions,
              testing::ElementsAreArray(expected_new_dimension_selection));
  TransformRep* new_rep = TransformAccess::rep(result_domain);
  if (TransformAccess::rep(original_domain) != new_rep) {
    EXPECT_EQ(0, new_rep->output_rank);
  }
}

/// Tests applying a `DimExpression` to a transform directly.
///
/// This tests both in place and out of place application of the
/// `DimExpression`, but does not test applying the `DimExpression` to the
/// identity transform over the domain of `original_transform`.  For that, see
/// the more comprehensive `TestDimExpression` below.
///
/// \param original_transform Initial transform
/// \param expression `DimExpression` object to apply
/// \param expected_new_dimension_selection Expected dimension selection after
///     applying transform
/// \param expected_new_transform Expected result of applying `expression` to
///     `original_transform`.
/// \param can_operate_in_place If `true`, `expression` is expected to operate
///     in place when applied to a transform with a reference count of 1.
template <typename OriginalTransform, typename Expression,
          typename ExpectedResult>
void TestDimExpressionOutOfPlaceAndInplace(
    const OriginalTransform& original_transform, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedResult& expected_new_transform, bool can_operate_in_place) {
  TestDimExpressionOutOfPlace(original_transform, expression,
                              expected_new_dimension_selection,
                              expected_new_transform);
  TestDimExpressionOutOfPlaceDomainOnly(original_transform.domain(), expression,
                                        expected_new_dimension_selection,
                                        expected_new_transform.domain());
  TestDimExpressionInplace(original_transform, expression,
                           expected_new_dimension_selection,
                           expected_new_transform, can_operate_in_place);
  TestDimExpressionInplaceDomainOnly(
      original_transform.domain(), expression, expected_new_dimension_selection,
      expected_new_transform.domain(), can_operate_in_place);
}

/// Tests that `equivalent_indices` specifies pairs of equivalent input indices
/// for two transforms.
///
/// \param transform_a First IndexTransform.
/// \param transform_b Second IndexTransform.
/// \param equivalent_indices Sequence of pairs `(a_indices, b_indices)` of
///     equivalent index vectors into `transform_a` and `transform_b`.
template <typename TransformA, typename TransformB>
void TestTransformEquivalentIndices(
    const TransformA& transform_a, const TransformB& transform_b,
    const EquivalentIndices& equivalent_indices) {
  for (const auto& pair : equivalent_indices) {
    std::vector<Index> out_a(transform_a.output_rank());
    std::vector<Index> out_b(transform_b.output_rank());
    ASSERT_EQ(absl::Status(), transform_a.TransformIndices(pair.first, out_a))
        << "Input=" << ::testing::PrintToString(pair.first);
    ASSERT_EQ(absl::Status(), transform_b.TransformIndices(pair.second, out_b))
        << "Input=" << ::testing::PrintToString(pair.second);
    EXPECT_EQ(out_a, out_b) << "Inputs=" << ::testing::PrintToString(pair);
  }
}

/// Tests the application of a `DimExpression` to a transform.
///
/// \param original_transform Initial transform
/// \param expression `DimExpression` object to apply
/// \param expected_new_dimension_selection Expected dimension selection after
///     applying transform
/// \param expected_identity_new_transform Expected result of applying
///     `expression` to an identity transform over the domain of
///     `original_transform`.
/// \param expected_new_transform Expected result of applying `expression` to
///     `original_transform`.
/// \param equivalent_indices Sequence of pairs
///     `(original_indices, new_indices)` of equivalent index vectors into
///     `original_transform` and `new_transform`, respectively.
/// \param can_operate_in_place If `true`, `expression` is expected to operate
///     in place when applied to a transform with a reference count of 1.
/// \param test_compose If `true`, tests that `expression` applied to an
///     identity transform over `original_transform.input_domain()` also results
///     in `expected_new_transform`.
template <typename OriginalTransform, typename Expression,
          typename ExpectedIdentityResult, typename ExpectedResult>
void TestDimExpression(
    const OriginalTransform& original_transform, const Expression& expression,
    std::vector<DimensionIndex> expected_new_dimension_selection,
    const ExpectedIdentityResult& expected_identity_new_transform,
    const ExpectedResult& expected_new_transform,
    const EquivalentIndices& equivalent_indices,
    bool can_operate_in_place = true, bool test_compose = true) {
  auto original_copy = MakeNewTransformCopy(original_transform);

  auto original_identity = IdentityTransformLike(original_transform);

  TestDimExpressionOutOfPlaceAndInplace(
      original_identity, expression, expected_new_dimension_selection,
      expected_identity_new_transform, can_operate_in_place);

  // Check composed result.  (Composition doesn't give the same result if the
  // domain is empty).
  if (test_compose && original_transform.domain().num_elements() != 0) {
    auto composed_transform =
        ComposeTransforms(original_transform, expected_identity_new_transform)
            .value();
    CheckSameTypes<decltype(composed_transform), ExpectedResult>();
    EXPECT_EQ(expected_new_transform, composed_transform);
  }

  TestDimExpressionOutOfPlaceAndInplace(
      original_transform, expression, expected_new_dimension_selection,
      expected_new_transform, can_operate_in_place);

  TestTransformEquivalentIndices(original_transform, expected_new_transform,
                                 equivalent_indices);
}

/// Tests that applying a `DimExpression` to `original_transform` results in the
/// expected error.
///
/// \param original_transform The original transform.
/// \param expression The `DimExpression` to apply.
/// \param error_code The expected error code.
/// \param message_pattern Regular expression specifying the expected error
///     message.
template <typename OriginalTransform, typename Expression>
void TestDimExpressionError(const OriginalTransform& original_transform,
                            const Expression& expression,
                            absl::StatusCode error_code,
                            const std::string& message_pattern) {
  auto original_copy = MakeNewTransformCopy(original_transform);

  // Check out-of-place implementation.
  auto result = expression(original_transform);
  EXPECT_THAT(result, tensorstore::MatchesStatus(error_code, message_pattern));

  EXPECT_EQ(original_copy, original_transform);

  // Check in-place implementation.
  auto inplace_result = expression(std::move(original_copy));
  EXPECT_THAT(inplace_result,
              tensorstore::MatchesStatus(error_code, message_pattern));
}

template <typename OriginalTransform, typename Expression,
          typename ExpectedDomain>
void TestDimExpressionErrorTransformOnly(
    const OriginalTransform& original_transform, const Expression& expression,
    absl::StatusCode error_code, const std::string& message_pattern,
    const ExpectedDomain& expected_domain) {
  TestDimExpressionError(original_transform, expression, error_code,
                         message_pattern);

  // Check out-of-place implementation.
  auto original_copy = MakeNewTransformCopy(original_transform);

  EXPECT_THAT(expression(original_transform.domain()),
              ::testing::Optional(expected_domain));

  // Check in-place implementation.
  EXPECT_THAT(expression(std::move(original_copy).domain()),
              ::testing::Optional(expected_domain));
}

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DIM_EXPRESSION_TESTUTIL_H_
