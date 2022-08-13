// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/index_space/dimension_units.h"

#include <optional>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/unit.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::DimensionUnitsToString;
using ::tensorstore::DimensionUnitsVector;
using ::tensorstore::MatchesStatus;
using ::tensorstore::MergeDimensionUnits;
using ::tensorstore::TransformInputDimensionUnits;
using ::tensorstore::TransformOutputDimensionUnits;
using ::tensorstore::Unit;

TEST(DimensionUnitsToStringTest, Basic) {
  EXPECT_EQ("[null, \"4 nm\"]", DimensionUnitsToString(DimensionUnitsVector{
                                    std::nullopt, Unit("4nm")}));
}

TEST(MergeDimensionUnitsTest, BothUnspecified) {
  DimensionUnitsVector existing_units{std::nullopt, std::nullopt};
  DimensionUnitsVector new_units{std::nullopt, std::nullopt};
  TENSORSTORE_EXPECT_OK(MergeDimensionUnits(existing_units, new_units));
  EXPECT_THAT(existing_units,
              ::testing::ElementsAre(std::nullopt, std::nullopt));
}

TEST(MergeDimensionUnitsTest, OneSpecifiedOneUnspecified) {
  DimensionUnitsVector existing_units{std::nullopt, Unit("4nm")};
  DimensionUnitsVector new_units{Unit("8nm"), std::nullopt};
  TENSORSTORE_EXPECT_OK(MergeDimensionUnits(existing_units, new_units));
  EXPECT_THAT(existing_units, ::testing::ElementsAre(Unit("8nm"), Unit("4nm")));
}

TEST(MergeDimensionUnitsTest, BothSpecifiedSame) {
  DimensionUnitsVector existing_units{Unit("8nm"), Unit("4nm")};
  DimensionUnitsVector new_units{Unit("8nm"), std::nullopt};
  TENSORSTORE_EXPECT_OK(MergeDimensionUnits(existing_units, new_units));
  EXPECT_THAT(existing_units, ::testing::ElementsAre(Unit("8nm"), Unit("4nm")));
}

TEST(MergeDimensionUnitsTest, BothSpecifiedDistinct) {
  DimensionUnitsVector existing_units{std::nullopt, Unit("4nm")};
  DimensionUnitsVector new_units{Unit("8nm"), Unit("5nm")};
  EXPECT_THAT(
      MergeDimensionUnits(existing_units, new_units),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot merge dimension units \\[\"8 nm\", \"5 nm\"\\] "
                    "and \\[null, \"4 nm\"\\]"));
  // Verify that `existing_units` remains unmodified.
  EXPECT_THAT(existing_units,
              ::testing::ElementsAre(std::nullopt, Unit("4nm")));
}

std::optional<Unit> MakeRandomUnit(absl::BitGenRef gen) {
  constexpr std::string_view kBaseUnits[] = {
      "",
      "nm",
      "um",
  };
  if (absl::Bernoulli(gen, 0.2)) return std::nullopt;
  // Use integer multipliers to avoid precision issues.
  const double multiplier = absl::Uniform<int>(gen, 5, 20);
  const auto base_unit =
      kBaseUnits[absl::Uniform<size_t>(gen, 0, std::size(kBaseUnits))];
  return Unit(multiplier, std::string(base_unit));
}

DimensionUnitsVector MakeRandomDimensionUnits(DimensionIndex rank,
                                              absl::BitGenRef gen) {
  DimensionUnitsVector units(rank);
  for (auto& unit : units) {
    unit = MakeRandomUnit(gen);
  }
  return units;
}

TEST(TransformOutputDimensionUnitsTest, InvertibleRoundTrip) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_TRANSFORM_DIMENSION_UNITS_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto domain = tensorstore::IndexDomain(box);
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain);
    auto output_units = MakeRandomDimensionUnits(domain.rank(), gen);
    auto input_units = TransformOutputDimensionUnits(transform, output_units);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inv_transform,
                                     InverseTransform(transform));
    EXPECT_THAT(TransformInputDimensionUnits(transform, input_units),
                ::testing::Optional(::testing::ElementsAreArray(output_units)));
    EXPECT_THAT(TransformOutputDimensionUnits(inv_transform, input_units),
                ::testing::ElementsAreArray(output_units));
  }
}

TEST(TransformOutputDimensionUnitsTest, StridedNonInvertibleRoundTrip) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_TRANSFORM_DIMENSION_UNITS_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto domain = tensorstore::IndexDomain(box);
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters p;
    p.max_stride = 4;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain, p);
    auto output_units = MakeRandomDimensionUnits(domain.rank(), gen);
    auto input_units = TransformOutputDimensionUnits(transform, output_units);
    EXPECT_THAT(TransformInputDimensionUnits(transform, input_units),
                ::testing::Optional(::testing::ElementsAreArray(output_units)));
  }
}

TEST(TransformInputDimensionUnitsTest, NoCorrespondingOutputDimension) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto transform,                //
      tensorstore::IndexTransformBuilder(1, 0).Finalize());
  DimensionUnitsVector input_units{"4nm"};
  EXPECT_THAT(TransformInputDimensionUnits(transform, input_units),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "No output dimension corresponds to "
                            "input dimension 0 with unit 4 nm"));
}

TEST(TransformOutputDimensionUnitsTest, NonUnique) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto transform,                //
      tensorstore::IndexTransformBuilder(2, 3)
          .output_single_input_dimension(0, 0)
          .output_single_input_dimension(1, 0)
          .output_single_input_dimension(2, 1)
          .Finalize());
  DimensionUnitsVector output_units{"4nm", "5nm", "6nm"};
  EXPECT_THAT(TransformOutputDimensionUnits(transform, output_units),
              ::testing::ElementsAre(std::nullopt, Unit("6nm")));
}

}  // namespace
