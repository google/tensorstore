// Copyright 2022 The TensorStore Authors
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

#include <assert.h>
#include <stdint.h>

#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/batch.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/downsample.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/spec.h"
#include "tensorstore/stack.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/unit.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesArray;
using ::tensorstore::MatchesJson;
using ::tensorstore::OpenMode;
using ::tensorstore::ReadWriteMode;
using ::tensorstore::span;
using ::tensorstore::StatusIs;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

::nlohmann::json GetRank1Length4ArrayDriver(int inclusive_min,
                                            int exclusive_max = -1) {
  if (exclusive_max == -1) {
    exclusive_max = inclusive_min + 4;
  }
  auto result = ::nlohmann::json{
      {"driver", "array"},
      {"array", {1, 2, 3, 4}},
      {"dtype", "int32"},
      {"transform",
       {
           {"input_inclusive_min", {inclusive_min}},
           {"input_exclusive_max", {exclusive_max}},
           {"output", {{{"input_dimension", 0}, {"offset", -inclusive_min}}}},
       }},
  };
  if (inclusive_min == 0) result["transform"].erase("output");
  return result;
}

::nlohmann::json GetRank1Length4N5Driver(int inclusive_min,
                                         int exclusive_max = -1) {
  if (exclusive_max == -1) {
    exclusive_max = inclusive_min + 4;
  }
  auto path = absl::StrCat("p", inclusive_min, "_", exclusive_max, "/");
  auto result = ::nlohmann::json{
      {"driver", "n5"},
      {"kvstore",
       {
           {"driver", "memory"},
           {"path", std::move(path)},
       }},
      {"metadata",
       {
           {"compression", {{"type", "raw"}}},
           {"dataType", "int32"},
           {"dimensions", {4}},
           {"blockSize", {4}},
       }},
      {"transform",
       {
           {"input_inclusive_min", {inclusive_min}},
           {"input_exclusive_max", {exclusive_max}},
           {"output", {{{"input_dimension", 0}, {"offset", -inclusive_min}}}},
       }},
  };
  if (inclusive_min == 0) result["transform"].erase("output");
  return result;
}

TEST(StackDriverTest, OpenAndResolveBounds) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
      {"schema", {{"dimension_units", {"2px"}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  EXPECT_EQ(1, store.rank());
  EXPECT_EQ(tensorstore::dtype_v<int32_t>, store.dtype());
  EXPECT_EQ(tensorstore::Box<1>({-3}, {9}), store.domain().box());
  EXPECT_THAT(
      store.dimension_units(),
      ::testing::Optional(::testing::ElementsAre(tensorstore::Unit("2px"))));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
  EXPECT_THAT(
      spec.ToJson(),
      ::testing::Optional(MatchesJson({
          {"driver", "stack"},
          {"dtype", "int32"},
          {"layers",
           ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                      GetRank1Length4ArrayDriver(0),
                                      GetRank1Length4ArrayDriver(3, 6)})},
          {"transform",
           {
               {"input_exclusive_max", {6}},
               {"input_inclusive_min", {-3}},
           }},
          {"schema",
           {
               {"dimension_units", {{2.0, "px"}}},
               {"domain", {{"exclusive_max", {6}}, {"inclusive_min", {-3}}}},
           }},
      })));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved,
                                   ResolveBounds(store).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(spec, resolved.spec());
  EXPECT_THAT(
      spec.ToJson(),
      ::testing::Optional(MatchesJson({
          {"driver", "stack"},
          {"dtype", "int32"},
          {"layers",
           ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                      GetRank1Length4ArrayDriver(0),
                                      GetRank1Length4ArrayDriver(3, 6)})},
          {"transform",
           {
               {"input_exclusive_max", {6}},
               {"input_inclusive_min", {-3}},
           }},
          {"schema",
           {
               {"dimension_units", {{2.0, "px"}}},
               {"domain", {{"exclusive_max", {6}}, {"inclusive_min", {-3}}}},
           }},
      })));
}

TEST(StackDriverTest, OpenWithDomain) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
      {"schema", {{"domain", {{"shape", {5}}}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  ASSERT_EQ(tensorstore::Box<1>({0}, {5}), store.domain().box());
}

TEST(StackDriverTest, Read) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                     tensorstore::Read(store).result());

    EXPECT_THAT(array, MatchesArray<int32_t>(span<const Index, 1>({-3}),
                                             {1, 2, 3, 1, 2, 3, 1, 2, 3}));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto array,
        tensorstore::Read<tensorstore::zero_origin>(store).result());

    EXPECT_THAT(array, MatchesArray<int32_t>({1, 2, 3, 1, 2, 3, 1, 2, 3}));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto array, (tensorstore::Read<tensorstore::zero_origin>(
                         store | tensorstore::AllDims().IndexArraySlice(
                                     tensorstore::MakeArray<Index>({-2, 1, 4})))
                         .result()));

    EXPECT_THAT(array, MatchesArray<int32_t>({2, 2, 2}));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto array, (tensorstore::Read<tensorstore::zero_origin>(
                         store | tensorstore::AllDims().IndexSlice({3}))
                         .result()));
    EXPECT_THAT(array, tensorstore::MatchesScalarArray<int32_t>(1));
  }
}

TEST(StackDriverTest, ReadSparse) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(2)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  // Cannot read everything
  EXPECT_THAT(
      tensorstore::Read<tensorstore::zero_origin>(store).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cell with origin={1} missing layer mapping")));

  // Can read the backed data.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto array, tensorstore::Read<tensorstore::zero_origin>(
                        store | tensorstore::AllDims().SizedInterval({2}, {3}))
                        .result());

    EXPECT_THAT(array, MatchesArray<int32_t>({1, 2, 3}));
  }
}

TEST(StackDriverTest, NoLayers) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, tensorstore::Spec::FromJson(
                     {{"driver", "stack"},
                      {"layers", ::nlohmann::json::array_t()},
                      {"dtype", "int32"},
                      {"schema", {{"domain", {{"shape", {2, 3}}}}}}}));
  EXPECT_EQ(OpenMode::unknown, spec.open_mode());
  EXPECT_THAT(
      spec.dimension_units(),
      ::testing::Optional(::testing::ElementsAre(std::nullopt, std::nullopt)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec).result());
  EXPECT_EQ(tensorstore::IndexDomain({2, 3}), store.domain());
  EXPECT_EQ(tensorstore::dtype_v<int32_t>, store.dtype());
  EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  EXPECT_THAT(
      store.dimension_units(),
      ::testing::Optional(::testing::ElementsAre(std::nullopt, std::nullopt)));
  TENSORSTORE_EXPECT_OK(store.schema());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec_no_domain,
      tensorstore::Spec::FromJson({{"driver", "stack"},
                                   {"layers", ::nlohmann::json::array_t()},
                                   {"dtype", "int32"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto empty_domain, spec_no_domain.domain());
  EXPECT_FALSE(empty_domain.valid());
  EXPECT_THAT(tensorstore::Open(spec_no_domain).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain must be specified")));
}

TEST(StackDriverTest, Rank0) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {{
            {"driver", "array"},
            {"array", 1},
            {"dtype", "uint32"},
            {"transform", {{"input_rank", 0}}},
        },
        {
            {"driver", "array"},
            {"array", 42},
            {"dtype", "uint32"},
            {"transform", {{"input_rank", 0}}},
        }}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read<>(store).result());
  auto expected = tensorstore::MakeScalarArray(static_cast<uint32_t>(42));
  EXPECT_EQ(array, expected);
}

TEST(StackDriverTest, WriteCreate) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::open_or_create).result());

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(
          tensorstore::MakeOffsetArray({-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
          store)
          .result());
}

TEST(StackDriverTest, WriteSparse) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(2)})},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::open_or_create).result());

  // Cannot write everything
  EXPECT_THAT(
      tensorstore::Write(
          tensorstore::MakeOffsetArray({-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
          store)
          .result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cell with origin={1} missing layer mapping")));

  // Can write the backed data.
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeOffsetArray({-3}, {9, 8, 7}),
                         store | tensorstore::AllDims().SizedInterval({2}, {3}))
          .result());
}

TEST(StackDriverTest, ReadWriteNonExistingLayers) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  // Sublayers are opened on read; they do not exist.
  EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(store).result(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Error opening \"n5\" driver: ")));

  // Sublayers are opened on write; they do not exist.
  EXPECT_THAT(tensorstore::Write(tensorstore::MakeOffsetArray(
                                     {-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
                                 store)
                  .result(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Error opening \"n5\" driver: ")));
}

TEST(StackDriverTest, Schema_MismatchedDtype) {
  auto a = GetRank1Length4N5Driver(0);
  a["dtype"] = "int64";

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a, GetRank1Length4N5Driver(3, 6)})},
      {"schema",
       {
           {"dtype", "int32"},
       }},
  };

  EXPECT_THAT(tensorstore::Open(json_spec).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Specified dtype (int32) does not match "
                                 "existing value (int64)")));

  json_spec.erase("schema");

  EXPECT_THAT(tensorstore::Open(json_spec).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Specified dtype (int32) does not match "
                                 "existing value (int64)")));
}

TEST(StackDriverTest, Schema_MismatchedRank) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a, GetRank1Length4N5Driver(3, 6)})},
      {"schema",
       {
           {"rank", 1},
       }},
  };

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Layer 0: Rank of 2 does not match existing rank of 1")));

  json_spec.erase("schema");

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Layer 1: Rank of 1 does not match existing rank of 2")));
}

TEST(StackDriverTest, Schema_HasCodec) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
      {"schema",
       {
           {"codec", {{"driver", "n5"}}},
       }},
  };

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("codec option not supported by \"stack\" driver")));
}

TEST(StackDriverTest, Schema_HasChunkLayout) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
      {"schema",
       {
           {"chunk_layout", {{"inner_order", {0, 1}}}},
       }},
  };

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("chunk layout option not supported by \"stack\" driver")));
}

TEST(StackDriverTest, Schema_HasFillValue) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
      {"schema",
       {
           {"fill_value", 42},
       }},
  };

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("fill value option not supported by \"stack\" driver")));
}

TEST(StackDriverTest, HasKvstore) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto kvstore_spec, tensorstore::kvstore::Spec::FromJson("memory://"));

  EXPECT_THAT(
      tensorstore::Open(json_spec, kvstore_spec).result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("kvstore option not supported by \"stack\" driver")));
}

TEST(StackDriverTest, Schema_DimensionUnitsInSchema) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
      {"schema", {{"dimension_units", {"4nm", "1nm"}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::open_or_create).result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("1nm"))));
}

TEST(StackDriverTest, Schema_DimensionUnitsInLayer) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
      {"schema", {{"dimension_units", {"1ft", "2ft"}}}},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::open_or_create).result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("1ft"), tensorstore::Unit("2ft"))));
}

TEST(StackDriverTest, Schema_MismatchedDimensionUnits) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
      {"schema", {{"dimension_units", {"1ft", "2ft"}}}},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({a})},
      {"schema", {{"dimension_units", {"4nm", nullptr}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::open_or_create).result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("2ft"))));
}

TEST(StackDriverTest, Schema_MismatchedLayerDimensionUnits) {
  ::nlohmann::json a{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
      {"schema", {{"dimension_units", {"1ft", "2ft"}}}},
  };

  ::nlohmann::json b{
      {"driver", "array"},
      {"array", {{1, 3, 3}, {4, 6, 6}}},
      {"dtype", "int32"},
      {"schema", {{"dimension_units", {"4nm", nullptr}}}},
  };

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", {a, b}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  std::nullopt, tensorstore::Unit("2ft"))));
}

TEST(StackDriverTest, SchemaDomain_MismatchedShape) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
      {"schema", {{"domain", {{"shape", {2, 2}}}}}},
  };

  EXPECT_THAT(
      tensorstore::Open(json_spec).result(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Layer 0: Rank of 1 does not match existing rank of 2")));
}

TEST(StackDriverTest, InvalidLayerMappingBug) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"dtype", "uint8"},
      {"layers",
       {
           {
               {"array", 0},
               {"driver", "array"},
               {"dtype", "uint8"},
               {"transform",
                {
                    {"input_exclusive_max", {10}},
                    {"input_inclusive_min", {0}},
                    {"output", ::nlohmann::json::array_t()},
                }},
           },
           {
               {"driver", "array"},
               {"dtype", "uint8"},
               {"array", 0},
               {"transform",
                {
                    {"input_exclusive_max", {4}},
                    {"input_inclusive_min", {0}},
                    {"output", ::nlohmann::json::array_t()},
                }},
           },
           {
               {"driver", "array"},
               {"dtype", "uint8"},
               {"array", 0},
               {"transform",
                {
                    {"input_exclusive_max", {8}},
                    {"input_inclusive_min", {4}},
                    {"output", ::nlohmann::json::array_t()},
                }},
           },
       }},
      {"schema",
       {
           {"domain", {{"exclusive_max", {10}}, {"inclusive_min", {0}}}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(json_spec).result());
  TENSORSTORE_ASSERT_OK(tensorstore::Read(store).result());
}

TEST(StackDriverTest, OpenMode_Open) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
           },
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_EQ(OpenMode::open, spec.open_mode());
}

TEST(StackDriverTest, SpecDimensionUnitsMatch) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"4nm"}}}},
               {"open", true},
               {"create", true},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"4nm"}}}},
               {"open", true},
               {"create", true},
           },
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_THAT(spec.dimension_units(),
              ::testing::Optional(::testing::ElementsAre("4nm")));
}

TEST(StackDriverTest, SpecDimensionUnitsMismatch) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"4nm"}}}},
               {"open", true},
               {"create", true},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"8nm"}}}},
               {"open", true},
               {"create", true},
           },
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_THAT(spec.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(std::nullopt)));
}

TEST(StackDriverTest, SpecDimensionUnitsInSchema) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"4nm"}}}},
               {"open", true},
               {"create", true},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"},
                 {"domain", {{"shape", {5}}}},
                 {"dimension_units", {"8nm"}}}},
               {"open", true},
               {"create", true},
           },
       }},
      {"schema", {{"dimension_units", {"7s"}}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_THAT(spec.dimension_units(),
              ::testing::Optional(::testing::ElementsAre("7s")));
}

TEST(StackDriverTest, OpenMode_OpenCreate) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
               {"open", true},
               {"create", true},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
               {"open", true},
               {"create", true},
           },
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_EQ(OpenMode::open_or_create, spec.open_mode());
}

TEST(StackDriverTest, OpenMode_Mismatch) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
               {"open", true},
               {"create", true},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
               {"open", true},
           },
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  EXPECT_EQ(OpenMode::unknown, spec.open_mode());
}

TEST(StackDriverTest, DomainUnspecified) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}}},
           },
       }},
  };
  EXPECT_THAT(tensorstore::Spec::FromJson(json_spec),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Domain must be specified")));
}

TEST(StackDriverTest, RankMismatch) {
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers",
       {
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}},
           },
           {
               {"driver", "zarr3"},
               {"kvstore", "memory://"},
               {"schema",
                {{"dtype", "int32"}, {"domain", {{"shape", {5, 2}}}}}},
           },
       }},
  };
  EXPECT_THAT(
      tensorstore::Spec::FromJson(json_spec),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Layer 1: Rank of 2 does not match existing rank of 1")));
}

// Checks that Context is bound properly to unopened (`Spec`) layers.
TEST(OverlayTest, Context) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({1, 2, 3, 4, 5}), base)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec_shifted, base_spec | tensorstore::Dims(0).TranslateBy(5));
  // Create stack from two `Spec` objects.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay({base_spec, base_spec_shifted}, context));
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int32_t>(
                    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})));
  }

  // Create stack from `TensorStore` and `Spec`.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base, base_spec_shifted}, context));
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int32_t>(
                    {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})));
  }
}

TEST(OverlayTest, Transaction) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({1, 2, 3, 4, 5}), base)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec_shifted, base_spec | tensorstore::Dims(0).TranslateBy(5));
  tensorstore::Transaction txn1(tensorstore::atomic_isolated);
  tensorstore::Transaction txn2(tensorstore::atomic_isolated);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_shifted,
                                   base | tensorstore::Dims(0).TranslateBy(5));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay({base_spec, base_spec_shifted}, context, txn1));
    EXPECT_EQ(txn1, store.transaction());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay({(base | txn1).value(), base_spec_shifted},
                             context, txn1));
    EXPECT_EQ(txn1, store.transaction());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base, base_shifted}, context, txn1));
    EXPECT_EQ(txn1, store.transaction());
    TENSORSTORE_ASSERT_OK(
        tensorstore::Write(
            tensorstore::MakeArray<int32_t>({10, 20, 30, 40, 50}),
            store | tensorstore::Dims(0).SizedInterval(0, 5))
            .result());
    // Uncommitted writes stay isolated inside `txn1` until commit.
    EXPECT_THAT(tensorstore::Read(base).result(),
                ::testing::Optional(MatchesArray<int32_t>({1, 2, 3, 4, 5})));
    EXPECT_THAT(
        tensorstore::Read(store | tensorstore::Dims(0).SizedInterval(0, 5))
            .result(),
        ::testing::Optional(MatchesArray<int32_t>({10, 20, 30, 40, 50})));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay({base, base_spec_shifted}, context, txn1));
    EXPECT_EQ(txn1, store.transaction());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({(base | txn1).value(), base_shifted},
                                         context, txn1));
    EXPECT_EQ(txn1, store.transaction());
  }

  EXPECT_THAT(tensorstore::Overlay({tensorstore::TensorStore<>{}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Layer is not valid")));
  EXPECT_THAT(tensorstore::Overlay({tensorstore::Spec{}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Layer is not valid")));
  EXPECT_THAT(tensorstore::Overlay({base, tensorstore::TensorStore<>{}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 1: Layer is not valid")));
  EXPECT_THAT(tensorstore::Stack({tensorstore::TensorStore<>{}}, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Layer is not valid")));
  EXPECT_THAT(tensorstore::Concat({tensorstore::TensorStore<>{}}, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Layer is not valid")));

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay(
            {(base | txn1).value(), (base_shifted | txn1).value()}, context));
    EXPECT_EQ(txn1, store.transaction());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay(
                        {(base | txn1).value(), (base_shifted | txn1).value()},
                        context, txn1));
    EXPECT_EQ(txn1, store.transaction());
  }

  EXPECT_THAT(
      tensorstore::Overlay({(base | txn1).value(), base_spec_shifted}, context),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 1: Transaction mismatch")));

  EXPECT_THAT(
      tensorstore::Overlay({(base | txn1).value(), base_shifted}, context),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 1: Transaction mismatch")));

  EXPECT_THAT(
      tensorstore::Overlay(
          {(base | txn1).value(), (base_shifted | txn2).value()}, context),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 1: Transaction mismatch")));

  EXPECT_THAT(tensorstore::Overlay(
                  {(base | txn1).value(), (base_shifted | txn1).value()},
                  context, txn2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Transaction mismatch")));
}

TEST(OverlayTest, NoLayers) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Overlay({}, tensorstore::dtype_v<int32_t>,
                                       tensorstore::Schema::Shape({2, 3})));
  EXPECT_EQ(tensorstore::IndexDomain({2, 3}), store.domain());
  EXPECT_EQ(tensorstore::dtype_v<int32_t>, store.dtype());
  EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  EXPECT_THAT(
      store.dimension_units(),
      ::testing::Optional(::testing::ElementsAre(std::nullopt, std::nullopt)));
  TENSORSTORE_EXPECT_OK(store.schema());

  EXPECT_THAT(tensorstore::Overlay({}, tensorstore::dtype_v<int32_t>),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Domain must be specified")));
}

TEST(OverlayTest, DimensionUnitsErrorPropagated) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::FromArray(tensorstore::MakeArray<float>({42.0f})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transformed_base,
      base | tensorstore::Dims(0).Stride(std::numeric_limits<Index>::max() / 2 +
                                         2));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled,
      tensorstore::Downsample(transformed_base, {2},
                              tensorstore::DownsampleMethod::kMean));
  EXPECT_THAT(downsampled.dimension_units(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      tensorstore::Overlay({downsampled}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Layer 0:")));
}

TEST(StackDriverTest, SchemaRankWithUnspecifiedLayerRank) {
  {
    ::nlohmann::json json_spec{
        {"driver", "stack"},
        {"layers",
         {{
             {"driver", "zarr3"},
             {"kvstore", "memory://"},
             {"schema", {{"dtype", "int32"}}},
         }}},
        {"schema", {{"rank", 2}}},
    };
    EXPECT_THAT(tensorstore::Spec::FromJson(json_spec),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Layer 0: Domain must be specified")));
  }

  // Positive case: `"cast"` over an `"n5"` layer without an explicit layer
  // transform initially has `schema.rank() == dynamic_rank` before options
  // propagate, while `GetEffectiveDomain` resolves its rank-1 domain `[0, 2)`.
  auto context = tensorstore::Context::Default();
  ::nlohmann::json n5_base{
      {"driver", "n5"},
      {"kvstore", {{"driver", "memory"}, {"path", "cast_rank/"}}},
      {"metadata",
       {
           {"compression", {{"type", "raw"}}},
           {"dataType", "int32"},
           {"dimensions", {2}},
           {"blockSize", {2}},
       }},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(n5_base, context, OpenMode::create).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({7, 8}), base_store)
          .result());

  ::nlohmann::json stack_spec_json{
      {"driver", "stack"},
      {"layers",
       {{
           {"driver", "cast"},
           {"dtype", "int64"},
           {"base", n5_base},
       }}},
      {"schema", {{"rank", 1}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto stack_spec, tensorstore::Spec::FromJson(stack_spec_json));
  EXPECT_EQ(1, stack_spec.rank());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto stack_store, tensorstore::Open(stack_spec, context).result());
  EXPECT_THAT(tensorstore::Read(stack_store).result(),
              ::testing::Optional(MatchesArray<int64_t>({7, 8})));

  // Without an explicit stack `schema.rank`,
  // `StackDriverSpec::GetDimensionUnits` and `Open` still derive the rank-1
  // dimension units vector from the layer.
  stack_spec_json.erase("schema");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto stack_spec_unconstrained,
      tensorstore::Spec::FromJson(stack_spec_json));
  EXPECT_THAT(stack_spec_unconstrained.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(std::nullopt)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto stack_store_unconstrained,
      tensorstore::Open(stack_spec_unconstrained, context).result());
  EXPECT_THAT(stack_store_unconstrained.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(std::nullopt)));

  // Mismatched explicit stack `schema.rank` is rejected when propagated to the
  // layer.
  stack_spec_json["schema"] = {{"rank", 2}};
  EXPECT_THAT(
      tensorstore::Spec::FromJson(stack_spec_json),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Layer 0:")));

  // Passing an untransformed `Spec` layer directly to `Overlay` resolves its
  // identity transform from `GetEffectiveDomain`, or rejects if unspecified.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto n5_spec,
                                   tensorstore::Spec::FromJson(n5_base));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto n5_overlay,
                                   tensorstore::Overlay({n5_spec}, context));
  EXPECT_THAT(tensorstore::Read(n5_overlay).result(),
              ::testing::Optional(MatchesArray<int32_t>({7, 8})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto zarr_no_domain,
      tensorstore::Spec::FromJson({{"driver", "zarr3"},
                                   {"kvstore", "memory://"},
                                   {"schema", {{"dtype", "int32"}}}}));
  EXPECT_THAT(tensorstore::Overlay({zarr_no_domain}, context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Domain must be specified")));
}

TEST(OverlayTest, ReadWriteMode) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({1, 2, 3, 4, 5}), base)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec_shifted, base_spec | tensorstore::Dims(0).TranslateBy(5));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_shifted,
                                   base | tensorstore::Dims(0).TranslateBy(5));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay({base_spec, base_spec_shifted}, context));
    EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base_spec, base_spec_shifted},
                                         context, ReadWriteMode::read));
    EXPECT_EQ(ReadWriteMode::read, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base_spec, base_spec_shifted},
                                         context, ReadWriteMode::write));
    EXPECT_EQ(ReadWriteMode::write, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base, base_shifted}, context));
    EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Overlay({base, base_shifted}, context,
                                         ReadWriteMode::read));
    EXPECT_EQ(ReadWriteMode::read, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay(
            {tensorstore::ModeCast(base, ReadWriteMode::read).value(),
             base_shifted},
            context));
    EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay(
            {tensorstore::ModeCast(base, ReadWriteMode::read).value(),
             tensorstore::ModeCast(base_shifted, ReadWriteMode::write).value()},
            context));
    EXPECT_EQ(ReadWriteMode::read_write, store.read_write_mode());
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Overlay(
            {tensorstore::ModeCast(base, ReadWriteMode::read).value(),
             tensorstore::ModeCast(base_shifted, ReadWriteMode::read).value()},
            context));
    EXPECT_EQ(ReadWriteMode::read, store.read_write_mode());
  }

  EXPECT_THAT(
      tensorstore::Overlay(
          {tensorstore::ModeCast(base, ReadWriteMode::read).value(),
           tensorstore::ModeCast(base_shifted, ReadWriteMode::read).value()},
          context, ReadWriteMode::write),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Write mode not supported")));
}

TEST(OverlayTest, DomainUnspecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson({{"driver", "zarr3"},
                                   {"kvstore", "memory://"},
                                   {"schema", {{"dtype", "int32"}}}}));
  EXPECT_THAT(tensorstore::Overlay({base_spec}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 0: Domain must be specified")));
}

TEST(OverlayTest, DomainRankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec1,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {2, 3}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec2,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {2}}}}}}}));
  EXPECT_THAT(
      tensorstore::Overlay({base_spec1, base_spec2}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               MatchesRegex("Layer 1: Layer domain .* of rank 1 does not match "
                            "layer 0 rank of 2")));
}

TEST(OverlayTest, Spec) {
  using array_t = ::nlohmann::json::array_t;
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_shifted,
                                   base | tensorstore::Dims(0).TranslateBy(5));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Overlay({base, base_shifted}, context));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   store.spec(tensorstore::MinimalSpec{true}));
  EXPECT_THAT(
      spec.ToJson(),
      ::testing::Optional(MatchesJson({
          {"driver", "stack"},
          {"layers",
           {
               {
                   {"driver", "zarr3"},
                   {"dtype", "int32"},
                   {"transform",
                    {{"input_exclusive_max", array_t(1, array_t{5})},
                     {"input_inclusive_min", array_t{0}}}},
                   {"kvstore", {{"driver", "memory"}}},
               },
               {
                   {"driver", "zarr3"},
                   {"dtype", "int32"},
                   {"transform",
                    {{"input_exclusive_max", array_t(1, array_t{10})},
                     {"input_inclusive_min", array_t{5}},
                     {"output", {{{"input_dimension", 0}, {"offset", -5}}}}}},
                   {"kvstore", {{"driver", "memory"}}},
               },
           }},
          {"dtype", "int32"},
          {"schema",
           {{"domain",
             {{"inclusive_min", array_t{0}}, {"exclusive_max", array_t{10}}}}}},
          {"transform",
           {{"input_exclusive_max", array_t{10}},
            {"input_inclusive_min", array_t{0}}}},
      })));
}

TEST(OverlayTest, OpenMode) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_shifted,
                                   base | tensorstore::Dims(0).TranslateBy(5));
  TENSORSTORE_ASSERT_OK(
      tensorstore::Overlay({base, base_shifted}, context, OpenMode::open));
  TENSORSTORE_ASSERT_OK(tensorstore::Overlay({base, base_shifted}, context,
                                             OpenMode::open_or_create));
  EXPECT_THAT(
      tensorstore::Overlay({base, base_shifted}, context, OpenMode::create),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 0: Open mode of create is not compatible with "
                         "already-open "
                         "layer")));
}

TEST(OverlayTest, RecheckCached) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto base_shifted,
                                   base | tensorstore::Dims(0).TranslateBy(5));
  TENSORSTORE_ASSERT_OK(tensorstore::Overlay({base, base_shifted}, context));
  EXPECT_THAT(
      tensorstore::Overlay({base, base_shifted}, context,
                           tensorstore::RecheckCachedData{false}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 0: Cannot specify cache rechecking options "
                         "with already-open layer")));
  EXPECT_THAT(
      tensorstore::Overlay({base, base_shifted}, context,
                           tensorstore::RecheckCachedMetadata{false}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Layer 0: Cannot specify cache rechecking options "
                         "with already-open layer")));
}

TEST(OverlayTest, MissingDtype) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec, tensorstore::Spec::FromJson(
                          {{"driver", "zarr3"},
                           {"kvstore", "memory://"},
                           {"schema", {{"domain", {{"shape", {5}}}}}}}));
  EXPECT_THAT(tensorstore::Overlay({base_spec}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("dtype must be specified")));
}

TEST(OverlayTest, DtypeMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec1,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://a/"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec2,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://b/"},
           {"schema", {{"dtype", "int64"}, {"domain", {{"shape", {5}}}}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec2_shifted,
      base_spec2 | tensorstore::Dims(0).TranslateBy(5));
  EXPECT_THAT(tensorstore::Overlay({base_spec1, base_spec2_shifted}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 1: Layer dtype of int64 does not match "
                                 "existing dtype of int32")));
}

TEST(StackTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({1, 2, 3, 4, 5}), base)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Stack({base_spec, base}, 0, context));
  EXPECT_EQ(tensorstore::IndexDomain({2, 5}), store.domain());
  EXPECT_THAT(tensorstore::Read(store).result(),
              ::testing::Optional(tensorstore::MakeArray<int32_t>(
                  {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}})));
}

TEST(StackTest, NoLayers) {
  EXPECT_THAT(
      tensorstore::Stack({}, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("At least one layer must be specified for stack")));
}

TEST(StackTest, MaxRank) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema",
            {{"dtype", "int32"},
             {"domain", {{"shape", ::nlohmann::json::array_t(32, 1)}}}}}}));
  EXPECT_THAT(tensorstore::Stack({base_spec, base_spec}, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("stack would exceed maximum rank of 32")));
}

TEST(ConcatTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson(
          {{"driver", "zarr3"},
           {"kvstore", "memory://"},
           {"schema", {{"dtype", "int32"}, {"domain", {{"shape", {5}}}}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({1, 2, 3, 4, 5}), base)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Concat({base_spec, base}, 0, context));
  EXPECT_THAT(tensorstore::Read(store).result(),
              ::testing::Optional(tensorstore::MakeArray<int32_t>(
                  {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})));
}

TEST(ConcatTest, DimensionLabel) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec,
      tensorstore::Spec::FromJson({{"driver", "zarr3"},
                                   {"kvstore", "memory://"},
                                   {"schema",
                                    {{"dtype", "int32"},
                                     {"domain",
                                      {
                                          {"shape", {1, 5}},
                                          {"labels", {"x", "y"}},
                                      }}}}}));
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base,
      tensorstore::Open(base_spec, OpenMode::create, context).result());
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeArray<int32_t>({{1, 2, 3, 4, 5}}),
                         base)
          .result());
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Concat({base_spec, base}, "x", context));
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int32_t>(
                    {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}})));
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Concat({base_spec, base}, "y", context));
    EXPECT_THAT(tensorstore::Read(store).result(),
                ::testing::Optional(tensorstore::MakeArray<int32_t>(
                    {{1, 2, 3, 4, 5, 1, 2, 3, 4, 5}})));
  }

  EXPECT_THAT(tensorstore::Concat({base_spec, base}, "z", context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Label \"z\" does not match one of ")));
}

TEST(ConcatTest, NoLayers) {
  EXPECT_THAT(
      tensorstore::Concat({}, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("At least one layer must be specified for concat")));
}

// When the concat_dimension is specified as a label, the dimension labels have
// to be resolved early using a different code path.  This tests that code path.
TEST(ConcatTest, DimensionLabelMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec1,
      tensorstore::Spec::FromJson({{"driver", "zarr3"},
                                   {"kvstore", "memory://"},
                                   {"schema",
                                    {{"dtype", "int32"},
                                     {"domain",
                                      {
                                          {"shape", {1, 5}},
                                          {"labels", {"x", "y"}},
                                      }}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_spec2,
      tensorstore::Spec::FromJson({{"driver", "zarr3"},
                                   {"kvstore", "memory://"},
                                   {"schema",
                                    {{"dtype", "int32"},
                                     {"domain",
                                      {
                                          {"shape", {1, 5}},
                                          {"labels", {"x", "z"}},
                                      }}}}}));
  EXPECT_THAT(tensorstore::Concat({base_spec1, base_spec2}, "x"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Layer 1: Mismatch in dimension 1: Dimension "
                                 "labels do not match")));
}

TEST(StackDriverTest, ConcatBatchRead) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {
              {"driver", "zarr3"},
              {"kvstore", {{"driver", "memory"}}},
              {"metadata",
               {
                   {"shape", {256, 1024}},
                   {"data_type", "uint16"},
                   {"chunk_grid",
                    {
                        {"name", "regular"},
                        {"configuration", {{"chunk_shape", {64, 1024}}}},
                    }},
               }},
          },
          context, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto concat_store,
                                   tensorstore::Concat({store}, 0));

  auto future = tensorstore::Read(concat_store, tensorstore::Batch::New());
  TENSORSTORE_EXPECT_OK(future);
}

TEST(StackDriverTest, SerializationRoundTrip) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(
                                       {
                                           {"driver", "zarr3"},
                                           {"kvstore", {{"driver", "memory"}}},
                                           {"metadata",
                                            {
                                                {"shape", {1}},
                                                {"data_type", "uint16"},
                                                {"fill_value", 43},
                                            }},
                                       },
                                       tensorstore::OpenMode::create)
                                       .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto arr, tensorstore::FromArray(tensorstore::MakeArray<uint16_t>({42})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto concat_store,
                                   tensorstore::Concat({store, arr}, 0));
  EXPECT_THAT(tensorstore::Read(concat_store).result(),
              ::testing::Optional(MatchesArray<uint16_t>({43, 42})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto concat_store_rt,
                                   SerializationRoundTrip(concat_store));
  EXPECT_THAT(tensorstore::Read(concat_store_rt).result(),
              ::testing::Optional(MatchesArray<uint16_t>({43, 42})));
}

}  // namespace
