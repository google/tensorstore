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

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/open.h"
#include "tensorstore/progress.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::CopyProgressFunction;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesArray;
using ::tensorstore::MatchesJson;
using ::tensorstore::ReadProgressFunction;
using ::tensorstore::span;

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
  auto path = absl::StrCat("p", inclusive_min, exclusive_max, "/");
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
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
      {"schema", {{"dimension_units", {"2px"}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

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
  auto context = tensorstore::Context::Default();
  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
      {"schema", {{"domain", {{"shape", {5}}}}}},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

  ASSERT_EQ(tensorstore::Box<1>({0}, {5}), store.domain().box());
}

TEST(StackDriverTest, Read) {
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(0),
                                            GetRank1Length4ArrayDriver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

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
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4ArrayDriver(-3),
                                            GetRank1Length4ArrayDriver(2)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

  // Cannot read everything
  EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(store).result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Read cell origin=.1. missing layer mapping.*"));

  // Can read the backed data.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto array, tensorstore::Read<tensorstore::zero_origin>(
                        store | tensorstore::AllDims().SizedInterval({2}, {3}))
                        .result());

    EXPECT_THAT(array, MatchesArray<int32_t>({1, 2, 3}));
  }
}

TEST(StackDriverTest, Rank0) {
  auto context = tensorstore::Context::Default();

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

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read<>(store).result());
  auto expected = tensorstore::MakeScalarArray(static_cast<uint32_t>(42));
  EXPECT_EQ(array, expected);
}

TEST(StackDriverTest, WriteCreate) {
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context,
                                    tensorstore::OpenMode::open_or_create)
                      .result());

  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(
          tensorstore::MakeOffsetArray({-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
          store)
          .result());
}

TEST(StackDriverTest, WriteSparse) {
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(2)})},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context,
                                    tensorstore::OpenMode::open_or_create)
                      .result());

  // Cannot write everything
  EXPECT_THAT(tensorstore::Write(tensorstore::MakeOffsetArray(
                                     {-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
                                 store)
                  .result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Write cell origin=.1. missing layer mapping.*"));

  // Can write the backed data.
  TENSORSTORE_ASSERT_OK(
      tensorstore::Write(tensorstore::MakeOffsetArray({-3}, {9, 8, 7}),
                         store | tensorstore::AllDims().SizedInterval({2}, {3}))
          .result());
}

TEST(StackDriverTest, ReadWriteNonExistingLayers) {
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(-3),
                                            GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, context).result());

  // Sublayers are opened on read; they do not exist.
  EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(store).result(),
              tensorstore::MatchesStatus(absl::StatusCode::kNotFound,
                                         ".*Error opening \"n5\" driver: .*"));

  // Sublayers are opened on write; they do not exist.
  EXPECT_THAT(tensorstore::Write(tensorstore::MakeOffsetArray(
                                     {-3}, {9, 8, 7, 6, 5, 4, 3, 2, 1}),
                                 store)
                  .result(),
              tensorstore::MatchesStatus(absl::StatusCode::kNotFound,
                                         ".*Error opening \"n5\" driver: .*"));
}

TEST(StackDriverTest, Schema_MismatchedDtype) {
  auto context = tensorstore::Context::Default();

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

  EXPECT_THAT(tensorstore::Open(json_spec, context).result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".*dtype .int32. does not match existing value .int64.*"));

  json_spec.erase("schema");

  EXPECT_THAT(tensorstore::Open(json_spec, context).result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  ".*dtype .int32. does not match existing value .int64.*"));
}

TEST(StackDriverTest, Schema_MismatchedRank) {
  auto context = tensorstore::Context::Default();

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
      tensorstore::Open(json_spec, context).result(),
      tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument,
                                 "Rank specified by rank .2. does not match "
                                 "existing rank specified by schema .1."));

  json_spec.erase("schema");

  EXPECT_THAT(
      tensorstore::Open(json_spec, context).result(),
      tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument,
                                 "Rank specified by rank .1. does not match "
                                 "existing rank specified by schema .2."));
}

TEST(StackDriverTest, Schema_HasCodec) {
  auto context = tensorstore::Context::Default();

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

  EXPECT_THAT(tensorstore::Open(json_spec, context).result(),
              tensorstore::MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "\"codec\" not supported by \"stack\" driver"));
}

TEST(StackDriverTest, Schema_DimensionUnitsInSchema) {
  auto context = tensorstore::Context::Default();

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
      auto store, tensorstore::Open(json_spec, context,
                                    tensorstore::OpenMode::open_or_create)
                      .result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("1nm"))));
}

TEST(StackDriverTest, Schema_DimensionUnitsInLayer) {
  auto context = tensorstore::Context::Default();

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
      auto store, tensorstore::Open(json_spec, context,
                                    tensorstore::OpenMode::open_or_create)
                      .result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("1ft"), tensorstore::Unit("2ft"))));
}

TEST(StackDriverTest, Schema_MismatchedDimensionUnits) {
  auto context = tensorstore::Context::Default();

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
      auto store, tensorstore::Open(json_spec, context,
                                    tensorstore::OpenMode::open_or_create)
                      .result());

  EXPECT_THAT(store.dimension_units(),
              ::testing::Optional(::testing::ElementsAre(
                  tensorstore::Unit("4nm"), tensorstore::Unit("2ft"))));
}

TEST(StackDriverTest, SchemaDomain_MismatchedShape) {
  auto context = tensorstore::Context::Default();

  ::nlohmann::json json_spec{
      {"driver", "stack"},
      {"layers", ::nlohmann::json::array_t({GetRank1Length4N5Driver(0),
                                            GetRank1Length4N5Driver(3, 6)})},
      {"schema", {{"domain", {{"shape", {2, 2}}}}}},
  };

  EXPECT_THAT(tensorstore::Open(json_spec, context).result(),
              tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument,
                                         "Rank specified by.*"));
}

}  // namespace
