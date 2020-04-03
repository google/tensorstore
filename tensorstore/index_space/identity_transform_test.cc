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

/// Tests for IdentityTransform.

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using tensorstore::AllocateArray;
using tensorstore::Box;
using tensorstore::IdentityTransform;
using tensorstore::Index;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::span;

TEST(IdentityTransformTest, Static) {
  auto t = IdentityTransform<2>();
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({1, 1})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IdentityTransformTest, Dynamic) {
  auto t = IdentityTransform(2);
  static_assert(std::is_same<decltype(t), IndexTransform<>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({1, 1})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IdentityTransformTest, LabeledCString) {
  auto t = IdentityTransform({"x", "y"});
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({1, 1})
                .input_labels({"x", "y"})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IdentityTransformTest, LabeledStdString) {
  auto t = IdentityTransform({std::string("x"), std::string("y")});
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({1, 1})
                .input_labels({"x", "y"})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IndexTransformTest, LabeledStringView) {
  auto t = IdentityTransform({absl::string_view("x"), absl::string_view("y")});
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({1, 1})
                .input_labels({"x", "y"})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IdentityTransformLikeTest, IndexTransform) {
  EXPECT_EQ((IndexTransformBuilder<2, 2>()
                 .input_origin({1, 2})
                 .input_shape({3, 4})
                 .implicit_lower_bounds({0, 1})
                 .implicit_upper_bounds({1, 0})
                 .input_labels({"x", "y"})
                 .output_single_input_dimension(0, 0, 1, 0)
                 .output_single_input_dimension(1, 0, 1, 1)
                 .Finalize()
                 .value()),
            IdentityTransformLike(IndexTransformBuilder<2, 3>()
                                      .input_origin({1, 2})
                                      .input_shape({3, 4})
                                      .implicit_lower_bounds({0, 1})
                                      .implicit_upper_bounds({1, 0})
                                      .input_labels({"x", "y"})
                                      .output_single_input_dimension(0, 5, 7, 1)
                                      .output_single_input_dimension(1, 6, 8, 0)
                                      .output_single_input_dimension(2, 7, 9, 0)
                                      .Finalize()
                                      .value()));
}

TEST(IdentityTransformLikeTest, Array) {
  EXPECT_EQ((IndexTransformBuilder<2, 2>()
                 .input_origin({0, 0})
                 .input_shape({3, 5})
                 .output_single_input_dimension(0, 0, 1, 0)
                 .output_single_input_dimension(1, 0, 1, 1)
                 .Finalize()
                 .value()),
            IdentityTransformLike(AllocateArray<float>({3, 5})));
}

TEST(IdentityTransformTest, StaticBox) {
  auto box = Box({1, 2}, {3, 4});
  auto t = IdentityTransform(box);
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_origin({1, 2})
                .input_shape({3, 4})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
  EXPECT_EQ(box, t.domain().box());
  static_assert(tensorstore::HasBoxDomain<IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(box, GetBoxDomainOf(t));
}

TEST(IdentityTransformTest, DynamicBox) {
  auto t = IdentityTransform(Box<>({1, 2}, {3, 4}));
  static_assert(std::is_same<decltype(t), IndexTransform<>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_origin({1, 2})
                .input_shape({3, 4})
                .output_single_input_dimension(0, 0, 1, 0)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value(),
            t);
}

TEST(IdentityTransformTest, FromShape) {
  auto t = IdentityTransform(span<const Index, 2>({2, 3}));
  static_assert(std::is_same<decltype(t), IndexTransform<2, 2>>::value, "");
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_origin({0, 0})
                .input_shape({2, 3})
                .output_single_input_dimension(0, 0)
                .output_single_input_dimension(1, 1)
                .Finalize()
                .value(),
            t);
}

}  // namespace
