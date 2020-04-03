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

/// Tests for PropagateInputDomainResizeToOutput.

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::DimensionIndex;
using tensorstore::IdentityTransform;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransformBuilder;
using tensorstore::IndexTransformView;
using tensorstore::kImplicit;
using tensorstore::kInfIndex;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StrCat;

TEST(ValidateInputDimensionResizeTest, ValidArguments) {
  using tensorstore::internal_index_space::ValidateInputDimensionResize;
  using OIII = tensorstore::OptionallyImplicitIndexInterval;

  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/kImplicit,
                /*requested_exclusive_max=*/kImplicit));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/1,
                /*requested_exclusive_max=*/1));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/kImplicit,
                /*requested_exclusive_max=*/1));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/1,
                /*requested_exclusive_max=*/kImplicit));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/-kInfIndex + 1,
                /*requested_exclusive_max=*/kImplicit));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/kImplicit,
                /*requested_exclusive_max=*/+kInfIndex));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{},
                /*requested_inclusive_min=*/kImplicit,
                /*requested_exclusive_max=*/-kInfIndex + 2));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{IndexInterval::UncheckedClosed(1, 10), false, false},
                /*requested_inclusive_min=*/kImplicit,
                /*requested_exclusive_max=*/kImplicit));
  EXPECT_EQ(Status(),
            ValidateInputDimensionResize(  //
                OIII{IndexInterval::UncheckedClosed(1, 10), true, true},
                /*requested_inclusive_min=*/-kInfIndex,
                /*requested_exclusive_max=*/+kInfIndex + 1));
}

TEST(ValidateInputDimensionResizeTest, InvalidArguments) {
  using tensorstore::internal_index_space::ValidateInputDimensionResize;
  using OIII = tensorstore::OptionallyImplicitIndexInterval;
  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{},
                  /*requested_inclusive_min=*/-kInfIndex - 1,
                  /*requested_exclusive_max=*/kImplicit),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            StrCat("Invalid requested inclusive min value ",
                                   -kInfIndex - 1)));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{},
                  /*requested_inclusive_min=*/kInfIndex,
                  /*requested_exclusive_max=*/kImplicit),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  StrCat("Invalid requested inclusive min value ", kInfIndex)));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{},
                  /*requested_inclusive_min=*/kImplicit,
                  /*requested_exclusive_max=*/-kInfIndex + 1),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            StrCat("Invalid requested exclusive max value ",
                                   -kInfIndex + 1)));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{},
                  /*requested_inclusive_min=*/kImplicit,
                  /*requested_exclusive_max=*/0x7ffffffffffffffe /*=2^63-1*/),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            StrCat("Invalid requested exclusive max value ",
                                   0x7ffffffffffffffe)));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{},
                  /*requested_inclusive_min=*/0,
                  /*requested_exclusive_max=*/-1),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid requested bounds \\[0, -1\\)"));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{IndexInterval::UncheckedClosed(1, 10), false, true},
                  /*requested_inclusive_min=*/0,
                  /*requested_exclusive_max=*/10),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot change explicit lower bound"));

  EXPECT_THAT(ValidateInputDimensionResize(  //
                  OIII{IndexInterval::UncheckedClosed(1, 10), true, false},
                  /*requested_inclusive_min=*/0,
                  /*requested_exclusive_max=*/10),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot change explicit upper bound"));
}

TEST(PropagateInputDomainResizeToOutputTest, ResizedSingleInputDimension) {
  auto transform = IndexTransformBuilder<>(1, 1)
                       .input_origin({1})
                       .input_exclusive_max({14})
                       .implicit_lower_bounds({1})
                       .implicit_upper_bounds({1})
                       .output_single_input_dimension(0, -5, 1, 0)
                       .Finalize()
                       .value();
  const auto test_resize = [&](bool can_resize_tied_bounds) {
    // `can_resize_tied_bounds` should have no effect in this case.

    Index requested_input_inclusive_min[] = {3};
    Index requested_input_exclusive_max[] = {10};
    Index output_inclusive_min_constraint[1];
    Index output_exclusive_max_constraint[1];
    Index new_output_inclusive_min[1];
    Index new_output_exclusive_max[1];
    bool is_noop;
    ASSERT_EQ(
        Status(),
        tensorstore::PropagateInputDomainResizeToOutput(
            transform, requested_input_inclusive_min,
            requested_input_exclusive_max, can_resize_tied_bounds,
            output_inclusive_min_constraint, output_exclusive_max_constraint,
            new_output_inclusive_min, new_output_exclusive_max, &is_noop));
    EXPECT_EQ(false, is_noop);

    EXPECT_THAT(output_inclusive_min_constraint,
                ::testing::ElementsAre(kImplicit));
    EXPECT_THAT(output_exclusive_max_constraint,
                ::testing::ElementsAre(kImplicit));

    EXPECT_THAT(new_output_inclusive_min, ::testing::ElementsAre(-5 + 1 * 3));
    EXPECT_THAT(new_output_exclusive_max, ::testing::ElementsAre(-5 + 1 * 10));
  };
  for (bool can_resize_tied_bounds : {false, true}) {
    test_resize(can_resize_tied_bounds);
  }
}

TEST(PropagateInputDomainResizeToOutputTest,
     ResizedSingleInputDimensionAndConstant) {
  auto transform = IndexTransformBuilder<>(1, 2)
                       .input_origin({1})
                       .input_exclusive_max({14})
                       .implicit_lower_bounds({1})
                       .implicit_upper_bounds({1})
                       .output_single_input_dimension(0, -5, 1, 0)
                       .output_constant(1, 10)
                       .Finalize()
                       .value();
  const auto test_resize = [&](bool can_resize_tied_bounds, bool resize_lower,
                               bool resize_upper) {
    Index requested_input_inclusive_min[] = {resize_lower ? 3 : kImplicit};
    Index requested_input_exclusive_max[] = {resize_upper ? 10 : kImplicit};
    Index output_inclusive_min_constraint[2];
    Index output_exclusive_max_constraint[2];
    Index new_output_inclusive_min[2];
    Index new_output_exclusive_max[2];
    bool is_noop;
    ASSERT_EQ(
        Status(),
        tensorstore::PropagateInputDomainResizeToOutput(
            transform, requested_input_inclusive_min,
            requested_input_exclusive_max, can_resize_tied_bounds,
            output_inclusive_min_constraint, output_exclusive_max_constraint,
            new_output_inclusive_min, new_output_exclusive_max, &is_noop));

    EXPECT_EQ(!resize_lower && !resize_upper, is_noop);

    EXPECT_THAT(
        output_inclusive_min_constraint,
        ::testing::ElementsAre(
            kImplicit, can_resize_tied_bounds || is_noop ? kImplicit : 10));
    EXPECT_THAT(
        output_exclusive_max_constraint,
        ::testing::ElementsAre(
            kImplicit, can_resize_tied_bounds || is_noop ? kImplicit : 11));

    EXPECT_THAT(new_output_inclusive_min,
                ::testing::ElementsAre(resize_lower ? -5 + 1 * 3 : kImplicit,
                                       kImplicit));
    EXPECT_THAT(new_output_exclusive_max,
                ::testing::ElementsAre(resize_upper ? -5 + 1 * 10 : kImplicit,
                                       kImplicit));
  };
  for (bool can_resize_tied_bounds : {false, true}) {
    for (const bool resize_lower : {false, true}) {
      for (const bool resize_upper : {false, true}) {
        test_resize(can_resize_tied_bounds, resize_lower, resize_upper);
      }
    }
  }
}

TEST(PropagateInputDomainResizeToOutputTest,
     ResizedSingleInputDimensionAndIndexArray) {
  auto transform =
      IndexTransformBuilder<>(2, 2)
          .input_origin({1, 0})
          .input_exclusive_max({14, 4})
          .implicit_lower_bounds({1, 0})
          .implicit_upper_bounds({1, 0})
          .output_single_input_dimension(0, -5, 1, 0)
          .output_index_array(1, 0, 1, MakeArray<Index>({{1, 2, 3, 4}}))
          .Finalize()
          .value();

  Index requested_input_inclusive_min[] = {3, kImplicit};
  Index requested_input_exclusive_max[] = {10, kImplicit};
  Index output_inclusive_min_constraint[2];
  Index output_exclusive_max_constraint[2];
  Index new_output_inclusive_min[2];
  Index new_output_exclusive_max[2];
  bool is_noop;
  ASSERT_EQ(
      Status(),
      tensorstore::PropagateInputDomainResizeToOutput(
          transform, requested_input_inclusive_min,
          requested_input_exclusive_max, /*can_resize_tied_bounds=*/true,
          output_inclusive_min_constraint, output_exclusive_max_constraint,
          new_output_inclusive_min, new_output_exclusive_max, &is_noop));

  EXPECT_EQ(false, is_noop);

  EXPECT_THAT(output_inclusive_min_constraint,
              ::testing::ElementsAre(kImplicit, kImplicit));
  EXPECT_THAT(output_exclusive_max_constraint,
              ::testing::ElementsAre(kImplicit, kImplicit));

  EXPECT_THAT(new_output_inclusive_min,
              ::testing::ElementsAre(-5 + 1 * 3, kImplicit));
  EXPECT_THAT(new_output_exclusive_max,
              ::testing::ElementsAre(-5 + 1 * 10, kImplicit));
}

TEST(PropagateInputDomainResizeToOutputTest,
     ResizedSingleInputDimensionAndOtherSingleInputDimension) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({1, 10})
                       .input_exclusive_max({7, 14})
                       .implicit_lower_bounds({0, 1})
                       .implicit_upper_bounds({0, 1})
                       .output_single_input_dimension(0, -5, 1, 1)
                       .output_single_input_dimension(1, 2, -1, 0)
                       .Finalize()
                       .value();
  const auto test_resize = [&](bool can_resize_tied_bounds, bool resize_lower,
                               bool resize_upper) {
    Index requested_input_inclusive_min[] = {kImplicit,
                                             resize_lower ? 3 : kImplicit};
    Index requested_input_exclusive_max[] = {kImplicit,
                                             resize_upper ? 10 : kImplicit};
    Index output_inclusive_min_constraint[2];
    Index output_exclusive_max_constraint[2];
    Index new_output_inclusive_min[2];
    Index new_output_exclusive_max[2];
    bool is_noop;
    ASSERT_EQ(
        Status(),
        tensorstore::PropagateInputDomainResizeToOutput(
            transform, requested_input_inclusive_min,
            requested_input_exclusive_max, can_resize_tied_bounds,
            output_inclusive_min_constraint, output_exclusive_max_constraint,
            new_output_inclusive_min, new_output_exclusive_max, &is_noop));
    EXPECT_EQ(!resize_lower && !resize_upper, is_noop);
    EXPECT_THAT(
        output_inclusive_min_constraint,
        ::testing::ElementsAre(kImplicit, can_resize_tied_bounds || is_noop
                                              ? kImplicit
                                              : 2 + -1 * (7 - 1)));
    EXPECT_THAT(
        output_exclusive_max_constraint,
        ::testing::ElementsAre(kImplicit, can_resize_tied_bounds || is_noop
                                              ? kImplicit
                                              : 2 + -1 * 1 + 1));

    EXPECT_THAT(new_output_inclusive_min,
                ::testing::ElementsAre(resize_lower ? -5 + 1 * 3 : kImplicit,
                                       kImplicit));
    EXPECT_THAT(new_output_exclusive_max,
                ::testing::ElementsAre(resize_upper ? -5 + 1 * 10 : kImplicit,
                                       kImplicit));
  };
  for (bool can_resize_tied_bounds : {false, true}) {
    SCOPED_TRACE(StrCat("can_resize_tied_bounds=", can_resize_tied_bounds));
    for (const bool resize_lower : {false, true}) {
      SCOPED_TRACE(StrCat("resize_lower=", resize_lower));
      for (const bool resize_upper : {false, true}) {
        SCOPED_TRACE(StrCat("resize_upper=", resize_upper));
        test_resize(can_resize_tied_bounds, resize_lower, resize_upper);
      }
    }
  }
}

TEST(PropagateInputDomainResizeToOutputTest, InvalidArguments) {
  using tensorstore::internal::GetLValue;
  bool is_noop;
  const auto test_resize = [&](bool can_resize_tied_bounds) {
    // `can_resize_tied_bounds` should have no effect on these cases.
    EXPECT_THAT(
        tensorstore::PropagateInputDomainResizeToOutput(
            IdentityTransform(1),
            /*requested_input_inclusive_min=*/span<const Index>({0}),
            /*requested_input_exclusive_max=*/span<const Index>({-1}),
            can_resize_tied_bounds,
            /*output_inclusive_min_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*output_exclusive_max_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*new_output_inclusive_min=*/GetLValue(std::array<Index, 1>()),
            /*new_output_exclusive_max=*/GetLValue(std::array<Index, 1>()),
            &is_noop),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      "Invalid resize request for input dimension 0: "
                      "Invalid requested bounds \\[0, -1\\)"));

    EXPECT_THAT(
        tensorstore::PropagateInputDomainResizeToOutput(
            IndexTransformBuilder<>(1, 2)
                .output_single_input_dimension(0, 0)
                .output_constant(1, kInfIndex)
                .Finalize()
                .value(),
            /*requested_input_inclusive_min=*/span<const Index>({1}),
            /*requested_input_exclusive_max=*/span<const Index>({10}),
            can_resize_tied_bounds,
            /*output_inclusive_min_constraint=*/
            GetLValue(std::array<Index, 2>()),
            /*output_exclusive_max_constraint=*/
            GetLValue(std::array<Index, 2>()),
            /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
            /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
            &is_noop),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            StrCat("Output dimension 1 has constant map with invalid offset ",
                   kInfIndex)));

    EXPECT_THAT(
        tensorstore::PropagateInputDomainResizeToOutput(
            IndexTransformBuilder<>(1, 1)
                .output_single_input_dimension(0, 0, 2, 0)
                .Finalize()
                .value(),
            /*requested_input_inclusive_min=*/span<const Index>({1}),
            /*requested_input_exclusive_max=*/span<const Index>({10}),
            can_resize_tied_bounds,
            /*output_inclusive_min_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*output_exclusive_max_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*new_output_inclusive_min=*/GetLValue(std::array<Index, 1>()),
            /*new_output_exclusive_max=*/GetLValue(std::array<Index, 1>()),
            &is_noop),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Output dimension 0 depends on resized input "
                             "dimension 0 with non-unit stride of 2")));

    EXPECT_THAT(
        tensorstore::PropagateInputDomainResizeToOutput(
            IndexTransformBuilder<>(1, 1)
                .output_single_input_dimension(0, kInfIndex, 1, 0)
                .Finalize()
                .value(),
            /*requested_input_inclusive_min=*/span<const Index>({1}),
            /*requested_input_exclusive_max=*/span<const Index>({10}),
            can_resize_tied_bounds,
            /*output_inclusive_min_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*output_exclusive_max_constraint=*/
            GetLValue(std::array<Index, 1>()),
            /*new_output_inclusive_min=*/GetLValue(std::array<Index, 1>()),
            /*new_output_exclusive_max=*/GetLValue(std::array<Index, 1>()),
            &is_noop),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      StrCat("Error propagating bounds for output dimension 0 "
                             "from requested bounds for input dimension 0: "
                             ".*")));
  };
  for (bool can_resize_tied_bounds : {false, true}) {
    test_resize(can_resize_tied_bounds);
  }

  EXPECT_THAT(
      tensorstore::PropagateInputDomainResizeToOutput(
          IndexTransformBuilder<>(2, 2)
              .input_origin({0, 0})
              .input_shape({5, 2})
              .implicit_lower_bounds({1, 0})
              .implicit_upper_bounds({1, 0})
              .output_single_input_dimension(0, 0)
              .output_index_array(1, 0, 1, MakeArray<Index>({{0, 1}}))
              .Finalize()
              .value(),
          /*requested_input_inclusive_min=*/span<const Index>({1, kImplicit}),
          /*requested_input_exclusive_max=*/
          span<const Index>({10, kImplicit}), /*can_resize_tied_bounds=*/false,
          /*output_inclusive_min_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*output_exclusive_max_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
          /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
          &is_noop),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    StrCat("Output dimension 1 has index array map but "
                           "`resize_tied_bounds` was not specified")));

  EXPECT_THAT(
      tensorstore::PropagateInputDomainResizeToOutput(
          IndexTransformBuilder<>(1, 2)
              .output_single_input_dimension(0, 0)
              .output_single_input_dimension(1, 0)
              .Finalize()
              .value(),
          /*requested_input_inclusive_min=*/span<const Index>({1}),
          /*requested_input_exclusive_max=*/span<const Index>({10}),
          /*can_resize_tied_bounds=*/false,
          /*output_inclusive_min_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*output_exclusive_max_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
          /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
          &is_noop),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    StrCat("Input dimension 0 corresponds to a diagonal but "
                           "`resize_tied_bounds` was not specified")));

  EXPECT_THAT(
      tensorstore::PropagateInputDomainResizeToOutput(
          IndexTransformBuilder<>(2, 2)
              .output_single_input_dimension(0, 0, 2, 1)
              .output_single_input_dimension(1, 0)
              .Finalize()
              .value(),
          /*requested_input_inclusive_min=*/span<const Index>({1, kImplicit}),
          /*requested_input_exclusive_max=*/span<const Index>({10, kImplicit}),
          /*can_resize_tied_bounds=*/false,
          /*output_inclusive_min_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*output_exclusive_max_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
          /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
          &is_noop),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    StrCat("Output dimension 0 depends on input dimension 1 "
                           "with non-unit stride of 2 but `resize_tied_bounds` "
                           "was not specified")));

  EXPECT_THAT(
      tensorstore::PropagateInputDomainResizeToOutput(
          IndexTransformBuilder<>(2, 2)
              .input_origin({1, 1})
              .input_shape({10, 10})
              .implicit_lower_bounds({1, 0})
              .implicit_upper_bounds({1, 0})
              .output_single_input_dimension(0, kInfIndex, 1, 1)
              .output_single_input_dimension(1, 0)
              .Finalize()
              .value(),
          /*requested_input_inclusive_min=*/span<const Index>({1, kImplicit}),
          /*requested_input_exclusive_max=*/span<const Index>({10, kImplicit}),
          /*can_resize_tied_bounds=*/false,
          /*output_inclusive_min_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*output_exclusive_max_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
          /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
          &is_noop),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    StrCat("Error propagating bounds for output dimension 0 "
                           "from existing bounds for input dimension 1: .*")));

  // TODO(jbms): If support for treating rank-0 index arrays like constant maps
  // is added, this test will also need to change.
  EXPECT_THAT(
      tensorstore::PropagateInputDomainResizeToOutput(
          IndexTransformBuilder<>(1, 2)
              .input_origin({1})
              .input_shape({10})
              .implicit_lower_bounds({1})
              .implicit_upper_bounds({1})
              .output_index_array(0, 0, 1, MakeArray<Index>({5}))
              .output_single_input_dimension(1, 0)
              .Finalize()
              .value(),
          /*requested_input_inclusive_min=*/span<const Index>({2}),
          /*requested_input_exclusive_max=*/span<const Index>({7}),
          /*can_resize_tied_bounds=*/false,
          /*output_inclusive_min_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*output_exclusive_max_constraint=*/
          GetLValue(std::array<Index, 2>()),
          /*new_output_inclusive_min=*/GetLValue(std::array<Index, 2>()),
          /*new_output_exclusive_max=*/GetLValue(std::array<Index, 2>()),
          &is_noop),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    StrCat("Output dimension 0 has index array map but "
                           "`resize_tied_bounds` was not specified")));
}

}  // namespace
