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

#include "tensorstore/index_space/transform_array_constraints.h"

#include <gtest/gtest.h>

namespace {
using tensorstore::ContiguousLayoutOrder;
using tensorstore::IterationConstraints;
using tensorstore::TransformArrayConstraints;

TEST(TransformArrayConstraintsTest, Basic) {
  EXPECT_TRUE(
      TransformArrayConstraints(ContiguousLayoutOrder::c).order_constraint());
  EXPECT_EQ(IterationConstraints(ContiguousLayoutOrder::c,
                                 tensorstore::skip_repeated_elements),
            TransformArrayConstraints(
                IterationConstraints(ContiguousLayoutOrder::c,
                                     tensorstore::skip_repeated_elements))
                .iteration_constraints());
  EXPECT_FALSE(TransformArrayConstraints(tensorstore::unspecified_order)
                   .order_constraint());
  EXPECT_EQ(tensorstore::skip_repeated_elements,
            TransformArrayConstraints(tensorstore::skip_repeated_elements,
                                      tensorstore::may_allocate)
                .repeated_elements_constraint());
  EXPECT_EQ(tensorstore::may_allocate,
            TransformArrayConstraints(tensorstore::skip_repeated_elements,
                                      tensorstore::may_allocate)
                .allocate_constraint());
  EXPECT_EQ(tensorstore::must_allocate,
            TransformArrayConstraints(tensorstore::skip_repeated_elements,
                                      tensorstore::must_allocate)
                .allocate_constraint());
  EXPECT_EQ(
      tensorstore::c_order,
      TransformArrayConstraints(tensorstore::c_order, tensorstore::may_allocate)
          .order_constraint()
          .order());
  EXPECT_EQ(tensorstore::fortran_order,
            TransformArrayConstraints(tensorstore::fortran_order,
                                      tensorstore::may_allocate)
                .order_constraint()
                .order());
  static_assert(
      11 == TransformArrayConstraints(tensorstore::fortran_order,
                                      tensorstore::include_repeated_elements,
                                      tensorstore::must_allocate)
                .value(),
      "");
  static_assert(
      3 == TransformArrayConstraints(tensorstore::fortran_order,
                                     tensorstore::include_repeated_elements)
               .value(),
      "");
  static_assert(
      10 == TransformArrayConstraints(tensorstore::c_order,
                                      tensorstore::include_repeated_elements,
                                      tensorstore::must_allocate)
                .value(),
      "");
  static_assert(
      8 == TransformArrayConstraints(tensorstore::include_repeated_elements,
                                     tensorstore::must_allocate)
               .value(),
      "");
  EXPECT_EQ(tensorstore::fortran_order,
            TransformArrayConstraints(tensorstore::fortran_order,
                                      tensorstore::include_repeated_elements,
                                      tensorstore::must_allocate)
                .order_constraint()
                .order());
}

}  // namespace
