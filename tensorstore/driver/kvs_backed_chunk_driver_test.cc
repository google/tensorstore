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

#include "tensorstore/driver/kvs_backed_chunk_driver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/driver/kvs_backed_chunk_driver_impl.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Box;
using tensorstore::Index;
using tensorstore::kImplicit;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_kvs_backed_chunk_driver::ValidateResizeConstraints;

using ISpan = tensorstore::span<const Index>;

TEST(ValidateResizeConstraintsTest, Success) {
  EXPECT_EQ(Status(),
            ValidateResizeConstraints(
                /*current_domain=*/Box({0, 0}, {4, 5}),
                /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                /*new_exclusive_max=*/ISpan({kImplicit, 6}),
                /*inclusive_min_constraint=*/ISpan({0, 0}),
                /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
                /*expand_only=*/false,
                /*shrink_only=*/false));

  EXPECT_EQ(Status(), ValidateResizeConstraints(
                          /*current_domain=*/Box({0, 0}, {4, 5}),
                          /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                          /*new_exclusive_max=*/ISpan({4, 6}),
                          /*inclusive_min_constraint=*/ISpan({0, 0}),
                          /*exclusive_max_constraint=*/ISpan({4, kImplicit}),
                          /*expand_only=*/false,
                          /*shrink_only=*/false));

  EXPECT_EQ(Status(),
            ValidateResizeConstraints(
                /*current_domain=*/Box({0, 0}, {4, 5}),
                /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                /*new_exclusive_max=*/ISpan({kImplicit, 6}),
                /*inclusive_min_constraint=*/ISpan({0, 0}),
                /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
                /*expand_only=*/true,
                /*shrink_only=*/false));

  EXPECT_EQ(Status(),
            ValidateResizeConstraints(
                /*current_domain=*/Box({0, 0}, {4, 5}),
                /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                /*new_exclusive_max=*/ISpan({kImplicit, 3}),
                /*inclusive_min_constraint=*/ISpan({0, 0}),
                /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
                /*expand_only=*/false,
                /*shrink_only=*/true));

  EXPECT_EQ(Status(),
            ValidateResizeConstraints(
                /*current_domain=*/Box({0, 0}, {4, 5}),
                /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                /*new_exclusive_max=*/ISpan({kImplicit, 5}),
                /*inclusive_min_constraint=*/ISpan({0, 0}),
                /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
                /*expand_only=*/true,
                /*shrink_only=*/true));

  EXPECT_EQ(Status(),
            ValidateResizeConstraints(
                /*current_domain=*/Box({0, 0}, {4, 5}),
                /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
                /*new_exclusive_max=*/ISpan({kImplicit, 5}),
                /*inclusive_min_constraint=*/ISpan({0, 0}),
                /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
                /*expand_only=*/true,
                /*shrink_only=*/true));
}

TEST(ValidateResizeConstraintsTest, Failure) {
  EXPECT_THAT(  //
      ValidateResizeConstraints(
          /*current_domain=*/Box({0, 0}, {4, 5}),
          /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
          /*new_exclusive_max=*/ISpan({kImplicit, 6}),
          /*inclusive_min_constraint=*/ISpan({0, 0}),
          /*exclusive_max_constraint=*/ISpan({5, kImplicit}),
          /*expand_only=*/false,
          /*shrink_only=*/false),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would also affect output dimension 0 "
                    "over the out-of-bounds interval \\[4, 5\\)"));

  EXPECT_THAT(  //
      ValidateResizeConstraints(
          /*current_domain=*/Box({0, 0}, {4, 5}),
          /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
          /*new_exclusive_max=*/ISpan({kImplicit, 6}),
          /*inclusive_min_constraint=*/ISpan({0, 0}),
          /*exclusive_max_constraint=*/ISpan({3, kImplicit}),
          /*expand_only=*/false,
          /*shrink_only=*/false),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Resize operation would also affect output dimension 0 over the "
          "interval \\[3, 4\\) but `resize_tied_bounds` was not specified"));

  EXPECT_THAT(  //
      ValidateResizeConstraints(
          /*current_domain=*/Box({0, 0}, {4, 5}),
          /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
          /*new_exclusive_max=*/ISpan({kImplicit, 6}),
          /*inclusive_min_constraint=*/ISpan({0, 0}),
          /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
          /*expand_only=*/false,
          /*shrink_only=*/true),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Resize operation would expand output dimension 1 from "
          "\\[0, 5\\) to \\[0, 6\\) but `shrink_only` was specified"));

  EXPECT_THAT(  //
      ValidateResizeConstraints(
          /*current_domain=*/Box({0, 0}, {4, 5}),
          /*new_inclusive_min=*/ISpan({kImplicit, kImplicit}),
          /*new_exclusive_max=*/ISpan({kImplicit, 4}),
          /*inclusive_min_constraint=*/ISpan({0, 0}),
          /*exclusive_max_constraint=*/ISpan({kImplicit, kImplicit}),
          /*expand_only=*/true,
          /*shrink_only=*/false),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Resize operation would shrink output dimension 1 from "
          "\\[0, 5\\) to \\[0, 4\\) but `expand_only` was specified"));
}

}  // namespace
