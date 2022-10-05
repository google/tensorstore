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

#include "tensorstore/schema.h"

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::CodecSpec;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dtype_v;
using ::tensorstore::dynamic_rank;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::kInfSize;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Schema;
using ::tensorstore::Unit;

TEST(SchemaTest, JsonRoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<Schema>({
      {
          {"dtype", "uint8"},
      },
      {
          {"dtype", "uint8"},
          {"fill_value", 5},
      },
      {
          {"rank", 3},
          {"dtype", "uint8"},
          {"domain",
           {{"labels", {"x", "y", "z"}},
            {"inclusive_min", {1, 2, 3}},
            {"exclusive_max", {5, 6, 7}}}},
          {"chunk_layout",
           {
               {"codec_chunk",
                {
                    {"elements_soft_constraint", 20},
                    {"aspect_ratio", {1, 2, 3}},
                    {"shape", {nullptr, 4, 5}},
                }},
               {"read_chunk",
                {
                    {"elements", 30},
                    {"aspect_ratio", {4, 5, 6}},
                    {"shape_soft_constraint", {6, nullptr, 7}},
                }},
               {"write_chunk",
                {
                    {"elements", 40},
                    {"aspect_ratio_soft_constraint", {7, 8, 9}},
                    {"shape", {8, 9, nullptr}},
                }},
               {"grid_origin", {nullptr, nullptr, 11}},
               {"inner_order_soft_constraint", {2, 0, 1}},
           }},
          {"codec",
           {
               {"driver", "zarr"},
               {"compressor", nullptr},
               {"filters", nullptr},
           }},
          {"fill_value", 5},
      },
      {
          {"rank", 3},
          {"dimension_units", {{4, "nm"}, {4, "nm"}, {30, "nm"}}},
      },
  });
}

TEST(SchemaTest, CompareRank) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"rank", 2}},
      {{"rank", 3}},
  });
}

TEST(SchemaTest, CompareDtype) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"dtype", "uint8"}},
      {{"dtype", "uint32"}},
  });
}

TEST(SchemaTest, CompareDomain) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"rank", 3},
       {"domain",
        {
            {"labels", {"a", "b", "c"}},
            {"inclusive_min", {0, 1, 2}},
            {"exclusive_max", {4, 5, 6}},
        }}},
      {{"rank", 3},
       {"domain",
        {
            {"labels", {"a", "b", "d"}},
            {"inclusive_min", {0, 1, 2}},
            {"exclusive_max", {4, 5, 6}},
        }}},
  });
}

TEST(SchemaTest, CompareLayout) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"rank", 3}, {"chunk_layout", {{"inner_order", {0, 1, 2}}}}},
      {{"rank", 3}, {"chunk_layout", {{"inner_order", {2, 1, 0}}}}},
      {{"rank", 3},
       {"chunk_layout", {{"inner_order_soft_constraint", {2, 1, 0}}}}},
  });
}

TEST(SchemaTest, CompareFillValue) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"fill_value", 42}},
      {{"fill_value", 43}},
      {{"fill_value", {2, 3}}},
  });
}

TEST(SchemaTest, CompareCodec) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"codec",
        {{"driver", "zarr"}, {"compressor", nullptr}, {"filters", nullptr}}}},
  });
}

TEST(SchemaTest, CompareDimensionUnits) {
  tensorstore::TestCompareDistinctFromJson<Schema>({
      ::nlohmann::json::object_t(),
      {{"rank", 3}, {"dimension_units", {{4, "nm"}, {5, "nm"}, {30, "nm"}}}},
      {{"rank", 3}, {"dimension_units", {{4, "nm"}, {5, "nm"}, {31, "nm"}}}},
  });
}

void TestApplyIndexTransformRandomInvertible(bool allow_new_dims) {
  constexpr size_t kNumIterations = 10;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto output_schema,
      Schema::FromJson({
          {"rank", 3},
          {"dtype", "uint8"},
          {"domain",
           {{"labels", {"x", "y", "z"}},
            {"inclusive_min", {1, 2, 3}},
            {"exclusive_max", {5, 6, 7}}}},
          {"chunk_layout",
           {
               {"codec_chunk",
                {
                    {"elements_soft_constraint", 20},
                    {"aspect_ratio", {1, 2, 3}},
                    {"shape", {nullptr, 4, 5}},
                }},
               {"read_chunk",
                {
                    {"elements", 30},
                    {"aspect_ratio", {4, 5, 6}},
                    {"shape_soft_constraint", {6, nullptr, 7}},
                }},
               {"write_chunk",
                {
                    {"elements", 40},
                    {"aspect_ratio_soft_constraint", {7, 8, 9}},
                    {"shape", {8, 9, nullptr}},
                }},
               {"grid_origin", {nullptr, nullptr, 11}},
               {"inner_order_soft_constraint", {2, 0, 1}},
           }},
          {"codec",
           {
               {"driver", "zarr"},
               {"compressor", nullptr},
               {"filters", nullptr},
           }},
          {"fill_value", 5},
          {"dimension_units", {{4, "nm"}, {5, "nm"}, {30, "nm"}}},
      }));
  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_SCHEMA_TEST_SEED")};
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
        transform_p;
    transform_p.new_dims_are_singleton = true;
    if (!allow_new_dims) {
      transform_p.max_new_dims = 0;
    }
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, output_schema.domain(), transform_p);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inverse_transform,
                                     InverseTransform(transform));
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    SCOPED_TRACE(tensorstore::StrCat("inverse_transform=", inverse_transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_schema,
                                     output_schema | transform);

    // If `transform` may contain new singleton input dimensions, then
    // `inverse_transform` may contain `constant` output index maps, which
    // aren't supported by `TransformInputSpaceSchema`.
    if (!allow_new_dims) {
      auto input_schema2 = output_schema;
      TENSORSTORE_ASSERT_OK(
          input_schema2.TransformInputSpaceSchema(inverse_transform));
      EXPECT_EQ(input_schema, input_schema2)
          << "output_schema=" << output_schema;
    }

    auto output_schema2 = input_schema;
    TENSORSTORE_ASSERT_OK(output_schema2.TransformInputSpaceSchema(transform));
    EXPECT_EQ(output_schema, output_schema2) << "input_schema=" << input_schema;

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_output_schema,
                                     input_schema | inverse_transform);
    EXPECT_EQ(output_schema, new_output_schema)
        << "input_schema=" << input_schema;
  }
}

TEST(SchemaTest, ApplyIndexTransformRandomInvertible) {
  TestApplyIndexTransformRandomInvertible(/*allow_new_dims=*/true);
}

TEST(SchemaTest, ApplyIndexTransformRandomInvertibleNoNewDims) {
  TestApplyIndexTransformRandomInvertible(/*allow_new_dims=*/false);
}

TEST(SchemaTest, RankMismatch) {
  tensorstore::TestJsonBinderFromJson<Schema>({
      {{
           {"domain",
            {
                {"inclusive_min", {0, 1}},
                {"exclusive_max", {3, 5}},
            }},
           {"chunk_layout", {{"inner_order", {0, 1, 2}}}},
       },
       MatchesStatus(
           absl::StatusCode::kInvalidArgument,
           "Error parsing object member \"chunk_layout\": "
           "Rank specified by chunk_layout \\(3\\) does not match existing "
           "rank specified by schema \\(2\\)")},
  });
}

TEST(SchemaTest, FillValueMismatch) {
  tensorstore::TestJsonBinderFromJson<Schema>({
      {{
           {"domain",
            {
                {"inclusive_min", {0, 1}},
                {"exclusive_max", {3, 5}},
            }},
           {"fill_value", {1, 2, 3}},
       },
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"fill_value\": "
                     "fill_value is incompatible with domain: "
                     "Cannot broadcast array of shape "
                     "\\{3\\} to target shape \\{3, 4\\}")},
      {{
           {"rank", 1},
           {"fill_value", {{1, 2, 3}, {4, 5, 6}}},
       },
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"fill_value\": "
                     "Invalid fill_value for rank 1: "
                     "\\{\\{1, 2, 3\\}, \\{4, 5, 6\\}\\}")},
  });
}

TEST(SchemaTest, ApplyIndexTransformNoRank) {
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_schema, schema | tensorstore::Dims(0, 1).TranslateBy(5));
  EXPECT_EQ(schema, new_schema);
}

TEST(SchemaTest, ApplyIndexTransformNoDomain) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dtype", "uint8"},
                       {"chunk_layout",
                        {
                            {"inner_order", {0, 1, 2}},
                            {"read_chunk", {{"shape", {4, 5, 6}}}},
                            {"grid_origin", {1, 2, 3}},
                        }},
                       {"fill_value", {1, 2, 3}},
                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_schema,
      schema | tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_new_schema, Schema::FromJson({
                                    {"dtype", "uint8"},
                                    {"chunk_layout",
                                     {
                                         {"inner_order", {2, 1, 0}},
                                         {"read_chunk", {{"shape", {6, 5, 4}}}},
                                         {"grid_origin", {8, 7, 6}},
                                     }},
                                    {"fill_value", {{{1}}, {{2}}, {{3}}}},
                                }));
  EXPECT_EQ(expected_new_schema, new_schema);
}

TEST(SchemaTest, ApplyIndexTransformUnknownRankNonScalarFillValue) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, Schema::FromJson({
                                                    {"dtype", "uint8"},
                                                    {"fill_value", {1, 2, 3}},
                                                }));
  EXPECT_THAT(schema | tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose(),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Cannot apply dimension expression to schema constraints of "
                  "unknown rank with non-scalar fill_value"));
}

TEST(SchemaTest, ApplyIndexTransformUnknownRankScalarFillValue) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, Schema::FromJson({
                                                    {"dtype", "uint8"},
                                                    {"fill_value", 42},
                                                }));
  EXPECT_THAT(schema | tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose(),
              ::testing::Optional(schema));
}

TEST(SchemaTest, ApplyIndexTransformKnownRankNullTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dtype", "uint8"},
                       {"chunk_layout",
                        {
                            {"inner_order", {2, 1, 0}},
                            {"read_chunk", {{"shape", {6, 5, 4}}}},
                            {"grid_origin", {8, 7, 6}},
                        }},
                       {"fill_value", 42},
                   }));
  EXPECT_THAT(schema | tensorstore::IndexTransform<>(),
              ::testing::Optional(schema));
}

TEST(SchemaTest, ApplyIndexTransformRankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dtype", "uint8"},
                       {"chunk_layout",
                        {
                            {"inner_order", {2, 1, 0}},
                            {"read_chunk", {{"shape", {6, 5, 4}}}},
                            {"grid_origin", {8, 7, 6}},
                        }},
                       {"fill_value", 42},
                   }));
  EXPECT_THAT(schema | tensorstore::IdentityTransform(2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot transform schema of rank 3 by index "
                            "transform of rank 2 -> 2"));
}

TEST(SchemaTest, ApplyIndexTransformDomainBoundsMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   Schema::FromJson({
                                       {"domain", {{"shape", {2, 3}}}},
                                   }));
  EXPECT_THAT(
      schema | tensorstore::IdentityTransform({4, 5}),
      MatchesStatus(
          absl::StatusCode::kOutOfRange,
          "Propagated bounds \\[0, 2\\), with size=2, for dimension 0 are "
          "incompatible with existing bounds \\[0, 4\\), with size=4.*"));
}

TEST(SchemaTest, ApplyIndexTransformUnknownRankNullTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   Schema::FromJson({
                                       {"dtype", "uint8"},
                                       {"fill_value", {42, 43, 44}},
                                   }));
  EXPECT_THAT(schema | tensorstore::IndexTransform<>(),
              ::testing::Optional(schema));
}

TEST(SchemaTest, ApplyIndexTransformDimensionUnits) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dimension_units", {"4nm", nullptr, "5nm"}},
                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_schema,
      Schema::FromJson({
          {"dimension_units", {nullptr, "8nm", nullptr, "5nm"}},
      }));
  EXPECT_THAT(
      schema | tensorstore::Dims(0).Stride(2) | tensorstore::Dims(0).AddNew(),
      ::testing::Optional(expected_schema));
}

TEST(SchemaTest, TransformInputSpaceSchemaDimensionUnits) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dimension_units", {"4nm", nullptr, "5nm"}},
                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_schema,
                                   Schema::FromJson({
                                       {"dimension_units", {"2nm", "5nm"}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   tensorstore::IdentityTransform(2) |
                                       tensorstore::Dims(0).Stride(2) |
                                       tensorstore::Dims(1).AddNew());
  TENSORSTORE_ASSERT_OK(schema.TransformInputSpaceSchema(transform));
  EXPECT_EQ(expected_schema, schema);
}

TEST(SchemaTest, TransformInputSpaceSchemaDimensionUnitsError) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   Schema::FromJson({
                                       {"dimension_units", {"4nm", "5nm"}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(1) | tensorstore::Dims(0).AddNew());
  EXPECT_THAT(schema.TransformInputSpaceSchema(transform),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error transforming dimension_units: "
                            "No output dimension corresponds to input "
                            "dimension 0 with unit 4 nm"));
}

TEST(SchemaTest, DtypeSet) {
  Schema schema;
  EXPECT_FALSE(schema.dtype().valid());
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<uint32_t>));
  EXPECT_EQ(dtype_v<uint32_t>, schema.dtype());
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<uint32_t>));
  EXPECT_EQ(dtype_v<uint32_t>, schema.dtype());
  EXPECT_THAT(schema.Set(dtype_v<int32_t>),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Specified dtype \\(int32\\) does not match "
                            "existing value \\(uint32\\)"));
  EXPECT_EQ(dtype_v<uint32_t>, schema.dtype());
  TENSORSTORE_ASSERT_OK(schema.Override(dtype_v<int32_t>));
  EXPECT_EQ(dtype_v<int32_t>, schema.dtype());
  TENSORSTORE_ASSERT_OK(schema.Override(tensorstore::DataType()));
  EXPECT_EQ(tensorstore::DataType(), schema.dtype());
}

TEST(SchemaTest, Rank) {
  Schema schema;
  EXPECT_EQ(dynamic_rank, schema.rank());
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::RankConstraint(dynamic_rank)));
  EXPECT_EQ(dynamic_rank, schema.rank());
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::RankConstraint(3)));
  EXPECT_EQ(3, schema.rank());
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::RankConstraint(3)));
  EXPECT_THAT(schema.Set(tensorstore::RankConstraint(2)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Rank specified by rank \\(2\\) does not match "
                            "existing rank specified by schema \\(3\\)"));
  EXPECT_EQ(3, schema.rank());
}

TEST(SchemaTest, Domain) {
  Schema schema;
  EXPECT_FALSE(schema.domain().valid());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain1,
                                   IndexDomainBuilder(3)
                                       .shape({2, 3, kInfSize})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain2,
      IndexDomainBuilder(3).shape({2, 3, 5}).labels({"x", "", "z"}).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain3, IndexDomainBuilder(3).labels({"x", "y", ""}).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_domain3,
                                   IndexDomainBuilder(3)
                                       .shape({2, 3, 5})
                                       .labels({"x", "y", "z"})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain4, IndexDomainBuilder(3).shape({1, 3, 5}).Finalize());
  TENSORSTORE_ASSERT_OK(schema.Set(domain1));
  EXPECT_EQ(domain1, schema.domain());
  TENSORSTORE_ASSERT_OK(schema.Set(domain1));
  EXPECT_EQ(domain1, schema.domain());
  TENSORSTORE_ASSERT_OK(schema.Set(domain2));
  EXPECT_EQ(domain2, schema.domain());
  TENSORSTORE_ASSERT_OK(schema.Set(domain3));
  EXPECT_EQ(expected_domain3, schema.domain());
  EXPECT_THAT(schema.Set(domain4),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge index domain .*: "
                            "Mismatch in dimension 0: "
                            "Upper bounds do not match"));
  EXPECT_EQ(expected_domain3, schema.domain());
}

TEST(SchemaTest, DomainAfterFillValue) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::FillValue(tensorstore::MakeArray({1, 2, 3}))));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain1, IndexDomainBuilder(3).shape({2, 3, 4}).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain2, IndexDomainBuilder(3).shape({2, 3, 3}).Finalize());
  EXPECT_THAT(
      schema.Set(domain1),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "domain is incompatible with fill_value: Cannot broadcast "
                    "array of shape \\{3\\} to target shape \\{2, 3, 4\\}"));
  EXPECT_FALSE(schema.domain().valid());
  TENSORSTORE_EXPECT_OK(schema.Set(domain2));
  EXPECT_EQ(domain2, schema.domain());
}

TEST(SchemaTest, OverrideDomainAfterFillValue) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::FillValue(tensorstore::MakeArray({1, 2, 3}))));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain1, IndexDomainBuilder(3).shape({2, 3, 4}).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain2, IndexDomainBuilder(3).shape({2, 3, 3}).Finalize());
  EXPECT_THAT(
      schema.Override(domain1),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "domain is incompatible with fill_value: Cannot broadcast "
                    "array of shape \\{3\\} to target shape \\{2, 3, 4\\}"));
  EXPECT_FALSE(schema.domain().valid());
  TENSORSTORE_EXPECT_OK(schema.Override(domain2));
  EXPECT_EQ(domain2, schema.domain());
  TENSORSTORE_EXPECT_OK(schema.Override(tensorstore::IndexDomain<>()));
  EXPECT_FALSE(schema.domain().valid());
}

TEST(SchemaTest, RankAfterFillValue) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(
      Schema::FillValue(tensorstore::MakeArray({{1, 2, 3}, {4, 5, 6}}))));
  EXPECT_THAT(
      schema.Set(tensorstore::RankConstraint(1)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank specified by rank \\(1\\) is incompatible with existing "
          "fill_value of shape \\{2, 3\\}"));
  EXPECT_THAT(schema.Set(tensorstore::RankConstraint(33)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Rank 33 is outside valid range \\[0, 32\\]"));
  EXPECT_EQ(dynamic_rank, schema.rank());
  TENSORSTORE_EXPECT_OK(schema.Set(tensorstore::RankConstraint(3)));
  EXPECT_EQ(3, schema.rank());
}

TEST(SchemaTest, FillValue) {
  Schema schema;
  EXPECT_FALSE(schema.fill_value().valid());
  auto fill_value1 = tensorstore::MakeScalarArray<uint32_t>(42);
  auto fill_value2 = tensorstore::MakeArray<uint32_t>({{1, 2, 3}});
  auto fill_value2_normalized = tensorstore::MakeArray<uint32_t>({1, 2, 3});
  TENSORSTORE_ASSERT_OK(schema.Set(Schema::FillValue(fill_value2)));
  EXPECT_EQ(fill_value2_normalized, schema.fill_value());
  EXPECT_THAT(
      schema.Set(Schema::FillValue(fill_value1)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Specified fill_value \\(42\\) does not "
                    "match existing value in schema \\(\\{1, 2, 3\\}\\)"));
  EXPECT_EQ(fill_value2_normalized, schema.fill_value());
  TENSORSTORE_ASSERT_OK(schema.Set(Schema::FillValue(fill_value2)));
  EXPECT_EQ(fill_value2_normalized, schema.fill_value());

  // Setting null fill value has no effect
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::FillValue(tensorstore::SharedArray<const void>())));
  EXPECT_EQ(fill_value2_normalized, schema.fill_value());
}

TEST(SchemaTest, DimensionUnits) {
  Schema schema;
  EXPECT_TRUE(schema.dimension_units().empty());
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::DimensionUnits({"4nm", std::nullopt, std::nullopt})));
  EXPECT_EQ(3, schema.rank());
  EXPECT_THAT(schema.dimension_units(),
              ::testing::ElementsAre(std::optional<Unit>("4nm"),
                                     std::optional<Unit>(std::nullopt),
                                     std::optional<Unit>(std::nullopt)));
  // Setting empty dimension units vector has no effect.
  TENSORSTORE_ASSERT_OK(schema.Set(Schema::DimensionUnits()));
  EXPECT_THAT(schema.dimension_units(),
              ::testing::ElementsAre(std::optional<Unit>("4nm"),
                                     std::optional<Unit>(std::nullopt),
                                     std::optional<Unit>(std::nullopt)));
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::DimensionUnits({std::nullopt, std::nullopt, "5nm"})));
  EXPECT_THAT(schema.dimension_units(),
              ::testing::ElementsAre(std::optional<Unit>("4nm"),
                                     std::optional<Unit>(std::nullopt),
                                     std::optional<Unit>("5nm")));
  // If there is a conflict, no changes are made.
  EXPECT_THAT(schema.Set(Schema::DimensionUnits({std::nullopt, "6nm", "7nm"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge dimension units .*"));
  EXPECT_THAT(schema.dimension_units(),
              ::testing::ElementsAre(std::optional<Unit>("4nm"),
                                     std::optional<Unit>(std::nullopt),
                                     std::optional<Unit>("5nm")));
}

TEST(SchemaTest, Codec) {
  Schema schema;
  EXPECT_FALSE(schema.codec().valid());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec1, CodecSpec::FromJson({
                                                    {"driver", "zarr"},
                                                    {"compressor", nullptr},
                                                }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec2, CodecSpec::FromJson({
                                                    {"driver", "zarr"},
                                                    {"filters", nullptr},
                                                }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec3, CodecSpec::FromJson({
                                                    {"driver", "zarr"},
                                                    {"compressor", nullptr},
                                                    {"filters", nullptr},
                                                }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec4, CodecSpec::FromJson({
                                                    {"driver", "n5"},
                                                }));
  TENSORSTORE_ASSERT_OK(schema.Set(codec1));
  EXPECT_EQ(codec1, schema.codec());
  TENSORSTORE_ASSERT_OK(schema.Set(codec2));
  EXPECT_EQ(codec3, schema.codec());
  EXPECT_THAT(schema.Set(codec4),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge codec spec .* with .*"));
  EXPECT_EQ(codec3, schema.codec());
}

TEST(SchemaTest, SetSchema) {
  Schema constraints;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
                       {"dtype", "uint8"},
                       {"domain",
                        {
                            {"inclusive_min", {0, 1, {2}}},
                            {"exclusive_max", {{3}, 4, 4}},
                            {"labels", {"x", "y", "z"}},
                        }},
                       {"codec",
                        {
                            {"driver", "zarr"},
                            {"compressor", nullptr},
                            {"filters", nullptr},
                        }},
                       {"fill_value", {{1, 2}, {3, 4}, {5, 6}}},
                       {"codec",
                        {
                            {"driver", "zarr"},
                            {"compressor", nullptr},
                            {"filters", nullptr},
                        }},
                       {"chunk_layout",
                        {
                            {"inner_order", {0, 1, 2}},
                            {"read_chunk", {{"shape", {4, 5, 6}}}},
                            {"grid_origin", {1, 2, 3}},
                        }},
                       {"dimension_units", {"4nm", "5nm", "30nm"}},
                   }));
  TENSORSTORE_ASSERT_OK(constraints.Set(schema));
  EXPECT_THAT(constraints.ToJson(tensorstore::IncludeDefaults{false}),
              ::testing::Optional(MatchesJson({
                  {"rank", 3},
                  {"dtype", "uint8"},
                  {"domain",
                   {
                       {"inclusive_min", {0, 1, {2}}},
                       {"exclusive_max", {{3}, 4, 4}},
                       {"labels", {"x", "y", "z"}},
                   }},
                  {"fill_value", {{1, 2}, {3, 4}, {5, 6}}},
                  {"codec",
                   {
                       {"driver", "zarr"},
                       {"compressor", nullptr},
                       {"filters", nullptr},
                   }},
                  {"chunk_layout",
                   {
                       {"inner_order", {0, 1, 2}},
                       {"read_chunk", {{"shape", {4, 5, 6}}}},
                       {"grid_origin", {1, 2, 3}},
                   }},
                  {"dimension_units", {{4, "nm"}, {5, "nm"}, {30, "nm"}}},
              })));
  EXPECT_EQ(Schema(schema), constraints);
}

tensorstore::Result<Box<>> ChooseReadWriteChunkGrid(
    ::nlohmann::json constraints, DimensionIndex rank) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto schema, Schema::FromJson(constraints));
  tensorstore::Box<> chunk_template(rank);
  TENSORSTORE_RETURN_IF_ERROR(
      tensorstore::internal::ChooseReadWriteChunkGrid(chunk_template, schema));
  return chunk_template;
}

TEST(ChooseReadWriteChunkGridTest, NoConstraints) {
  EXPECT_THAT(
      ChooseReadWriteChunkGrid(
          {{"chunk_layout", {{"chunk", {{"elements", 64 * 64 * 64}}}}}}, 3),
      ::testing::Optional(Box({64, 64, 64})));
}

TEST(SchemaSerializationTest, SerializationRoundTrip) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto schema,                   //
      tensorstore::Schema::FromJson({
          {"rank", 3},
          {"dtype", "uint8"},
          {"domain",
           {{"labels", {"x", "y", "z"}},
            {"inclusive_min", {1, 2, 3}},
            {"exclusive_max", {5, 6, 7}}}},
          {"chunk_layout",
           {
               {"codec_chunk",
                {
                    {"elements_soft_constraint", 20},
                    {"aspect_ratio", {1, 2, 3}},
                    {"shape", {nullptr, 4, 5}},
                }},
               {"read_chunk",
                {
                    {"elements", 30},
                    {"aspect_ratio", {4, 5, 6}},
                    {"shape_soft_constraint", {6, nullptr, 7}},
                }},
               {"write_chunk",
                {
                    {"elements", 40},
                    {"aspect_ratio_soft_constraint", {7, 8, 9}},
                    {"shape", {8, 9, nullptr}},
                }},
               {"grid_origin", {nullptr, nullptr, 11}},
               {"inner_order_soft_constraint", {2, 0, 1}},
           }},
          {"codec",
           {
               {"driver", "zarr"},
               {"compressor", nullptr},
               {"filters", nullptr},
           }},
          {"fill_value", 5},
      }));
  tensorstore::serialization::TestSerializationRoundTrip(schema);
}

}  // namespace
