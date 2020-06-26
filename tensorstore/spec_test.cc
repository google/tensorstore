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

#include "tensorstore/spec.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::nlohmann::json;
using tensorstore::BoxView;
using tensorstore::DataType;
using tensorstore::DataTypeOf;
using tensorstore::DimensionIndex;
using tensorstore::dynamic_rank;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformSpec;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::Spec;
using tensorstore::StaticDataType;
using tensorstore::StaticRankCast;
using tensorstore::StrCat;

TEST(SpecTest, ToJson) {
  ::nlohmann::json spec_json({{"driver", "array"},
                              {"dtype", "int32"},
                              {"array", {1, 2, 3}},
                              {"rank", 1}});
  Spec spec =
      Spec::FromJson(spec_json, tensorstore::AllowUnregistered{true}).value();
  EXPECT_THAT(spec.ToJson(), ::testing::Optional(spec_json));
}

TEST(SpecTest, Comparison) {
  Spec spec_a = Spec::FromJson({{"driver", "array"},
                                {"dtype", "int32"},
                                {"array", {1, 2, 3}},
                                {"rank", 1}},
                               tensorstore::AllowUnregistered{true})
                    .value();
  Spec spec_b = Spec::FromJson({{"driver", "array"},
                                {"dtype", "int32"},
                                {"array", {1, 2, 3, 4}},
                                {"rank", 1}},
                               tensorstore::AllowUnregistered{true})
                    .value();
  EXPECT_EQ(spec_a, spec_a);
  EXPECT_NE(spec_a, spec_b);
}

TEST(SpecTest, ApplyIndexTransform) {
  // Tests successfully applying a DimExpression to a Spec.
  auto spec_with_transform =
      Spec::FromJson(
          ::nlohmann::json({{"driver", "array"},
                            {"dtype", "int32"},
                            {"array",
                             {
                                 {1, 2, 3, 4},
                                 {5, 6, 7, 8},
                                 {9, 10, 11, 12},
                             }},
                            {"transform",
                             {{"input_inclusive_min", {2, 4}},
                              {"input_shape", {3, 2}},
                              {"output",
                               {{{"input_dimension", 0}, {"offset", -2}},
                                {{"input_dimension", 1}, {"offset", -4}}}}}}}),
          tensorstore::AllowUnregistered{true})
          .value();
  EXPECT_EQ(2, spec_with_transform.rank());
  EXPECT_THAT(
      ChainResult(Spec::FromJson(
                      ::nlohmann::json(
                          {{"driver", "array"},
                           {"dtype", "int32"},
                           {"array",
                            {
                                {1, 2, 3, 4},
                                {5, 6, 7, 8},
                                {9, 10, 11, 12},
                            }},
                           {"transform",
                            {{"input_inclusive_min", {2, 4}},
                             {"input_shape", {3, 4}},
                             {"output",
                              {{{"input_dimension", 0}, {"offset", -2}},
                               {{"input_dimension", 1}, {"offset", -4}}}}}}}),
                      tensorstore::AllowUnregistered{true}),
                  tensorstore::Dims(1).SizedInterval(4, 2)),
      ::testing::Optional(spec_with_transform));

  // Tests applying a DimExpression to a Spec with unknown rank (fails).
  Spec spec_without_transform =
      Spec::FromJson(::nlohmann::json({{"driver", "array"},
                                       {"dtype", "int32"},
                                       {"array",
                                        {
                                            {1, 2, 3, 4},
                                            {5, 6, 7, 8},
                                            {9, 10, 11, 12},
                                        }}}),
                     tensorstore::AllowUnregistered{true})
          .value();
  EXPECT_THAT(ChainResult(spec_without_transform,
                          tensorstore::Dims(1).SizedInterval(4, 2)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform is unspecified"));
}

TEST(SpecTest, PrintToOstream) {
  ::nlohmann::json spec_json({{"driver", "array"},
                              {"dtype", "int32"},
                              {"array", {1, 2, 3}},
                              {"rank", 1}});
  Spec spec =
      Spec::FromJson(spec_json, tensorstore::AllowUnregistered{true}).value();
  EXPECT_EQ(spec_json.dump(), tensorstore::StrCat(spec));
}

}  // namespace
