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
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using tensorstore::ChunkLayout;
using tensorstore::DimensionIndex;
using tensorstore::dtype_v;
using tensorstore::dynamic_rank;
using tensorstore::Index;
using tensorstore::MatchesStatus;
using tensorstore::Schema;

TEST(SchemaTest, DefaultConstruct) {
  Schema schema;
  EXPECT_FALSE(schema.dtype().valid());
  EXPECT_EQ(dynamic_rank, schema.rank());
  EXPECT_FALSE(schema.domain().valid());
  EXPECT_FALSE(schema.chunk_layout().valid());
  EXPECT_FALSE(schema.codec().valid());
  EXPECT_FALSE(schema.fill_value().valid());
}

TEST(SchemaTest, DtypeOnly) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   Schema::FromJson({{"dtype", "uint8"}}));
  EXPECT_EQ(dtype_v<uint8_t>, schema.dtype());
  EXPECT_EQ(dynamic_rank, schema.rank());
  EXPECT_FALSE(schema.domain().valid());
  EXPECT_FALSE(schema.chunk_layout().valid());
  EXPECT_FALSE(schema.codec().valid());
  EXPECT_FALSE(schema.fill_value().valid());
}

TEST(SchemaTest, LayoutOnly) {
  ::nlohmann::json chunk_layout_json{
      {"inner_order", {0, 1}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto chunk_layout,
                                   ChunkLayout::FromJson(chunk_layout_json));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({{"chunk_layout", chunk_layout_json}}));
  EXPECT_FALSE(schema.dtype().valid());
  EXPECT_EQ(2, schema.rank());
  EXPECT_FALSE(schema.domain().valid());
  EXPECT_EQ(chunk_layout, schema.chunk_layout());
  EXPECT_FALSE(schema.codec().valid());
  EXPECT_FALSE(schema.fill_value().valid());
}

TEST(SchemaTest, DomainOnly) {
  ::nlohmann::json domain_json{
      {"inclusive_min", {0, 1}},
      {"exclusive_max", {3, 5}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain,
                                   tensorstore::ParseIndexDomain(domain_json));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema,
                                   Schema::FromJson({{"domain", domain_json}}));
  EXPECT_FALSE(schema.dtype().valid());
  EXPECT_EQ(2, schema.rank());
  EXPECT_EQ(domain, schema.domain());
  EXPECT_FALSE(schema.chunk_layout().valid());
  EXPECT_FALSE(schema.codec().valid());
  EXPECT_FALSE(schema.fill_value().valid());
}

TEST(SchemaTest, JsonRoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<Schema>(
      {
          {
              {"dtype", "uint8"},
          },
          {
              {"dtype", "uint8"},
              {"domain",
               {
                   {"inclusive_min", {0, 1}},
                   {"exclusive_max", {3, 5}},
               }},
              {"chunk_layout", {{"inner_order", {0, 1}}}},
              {"codec",
               {
                   {"driver", "zarr"},
                   {"compressor", nullptr},
               }},
              {"fill_value", 42},
          },
      },
      tensorstore::internal_json_binding::DefaultBinder<>,
      tensorstore::IncludeDefaults{false});
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
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Rank of chunk_layout \\(3\\) does not match rank of "
                     "domain \\(2\\)")},
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
                     "Invalid fill_value: Cannot broadcast array of shape "
                     "\\{3\\} to target shape \\{3, 4\\}")},
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
                            {"grid_origin", {1, 2, 3}},
                            {"write_chunk", {{"shape", {4, 5, 6}}}},
                        }},
                       {"fill_value", {1, 2, 3}},
                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_schema,
      schema | tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_new_schema,
      Schema::FromJson({
          {"dtype", "uint8"},
          {"chunk_layout",
           {
               {"inner_order", {2, 1, 0}},
               {"grid_origin", {8, 7, 6}},
               {"write_chunk", {{"shape", {6, 5, 4}}}},
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
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot apply dimension expression to schema of "
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
                            {"grid_origin", {8, 7, 6}},
                            {"write_chunk", {{"shape", {6, 5, 4}}}},
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
                            {"grid_origin", {8, 7, 6}},
                            {"write_chunk", {{"shape", {6, 5, 4}}}},
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
  EXPECT_THAT(schema | tensorstore::IdentityTransform({4, 5}),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Propagated bounds \\[0, 2\\) for dimension 0 are "
                            "incompatible with existing bounds \\[0, 4\\)\\."));
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

TEST(SchemaTest, ApplyIndexTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema, Schema::FromJson({
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
                            {"grid_origin", {1, 2, 3}},
                            {"write_chunk", {{"shape", {4, 5, 6}}}},
                        }},
                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_schema,
      schema |
          tensorstore::Dims(2, 1, 0).TranslateBy({10, 20, 30}).Transpose());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_schema,
      Schema::FromJson({
          {"dtype", "uint8"},
          {"domain",
           {
               {"inclusive_min", {{12}, 21, 30}},
               {"exclusive_max", {14, 24, {33}}},
               {"labels", {"z", "y", "x"}},
           }},
          {"codec",
           {
               {"driver", "zarr"},
               {"compressor", nullptr},
               {"filters", nullptr},
           }},
          {"fill_value", {{{1}, {3}, {5}}, {{2}, {4}, {6}}}},
          {"codec",
           {
               {"driver", "zarr"},
               {"compressor", nullptr},
               {"filters", nullptr},
           }},
          {"chunk_layout",
           {
               {"inner_order", {2, 1, 0}},
               {"grid_origin", {13, 22, 31}},
               {"write_chunk", {{"shape", {6, 5, 4}}}},
           }},
      }));
  EXPECT_EQ(expected_schema, new_schema);
}

TEST(SchemaTest, ApplyIndexTransformRandomInvertible) {
  constexpr size_t kNumIterations = 10;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto output_schema, Schema::FromJson({
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
                                   {"grid_origin", {1, 2, 3}},
                                   {"write_chunk", {{"shape", {4, 5, 6}}}},
                               }},
                          }));
  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_SCHEMA_TEST_SEED")};
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
        transform_p;
    transform_p.new_dims_are_singleton = false;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, output_schema.domain(), {});
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_schema,
                                     output_schema | transform);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inverse_transform,
                                     InverseTransform(transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_output_schema,
                                     input_schema | inverse_transform);
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    SCOPED_TRACE(tensorstore::StrCat("inverse_transform=", inverse_transform));
    EXPECT_EQ(output_schema, new_output_schema)
        << "input_schema=" << input_schema;
  }
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
      {{"domain",
        {
            {"labels", {"a", "b", "c"}},
            {"inclusive_min", {0, 1, 2}},
            {"exclusive_max", {4, 5, 6}},
        }}},
      {{"domain",
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
      {{"chunk_layout", {{"inner_order", {0, 1, 2}}}}},
      {{"chunk_layout", {{"inner_order", {2, 1, 0}}}}},
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

}  // namespace
