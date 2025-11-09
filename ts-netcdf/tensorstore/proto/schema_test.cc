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

#include "tensorstore/proto/schema.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/proto/schema.pb.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ParseSchemaFromProto;
using ::tensorstore::Schema;

template <typename Proto>
Proto ParseProtoOrDie(const std::string& asciipb) {
  return protobuf_matchers::internal::MakePartialProtoFromAscii<Proto>(asciipb);
}

auto DoEncode(const Schema& schema) {
  ::tensorstore::proto::Schema proto;
  ::tensorstore::EncodeToProto(proto, schema);
  return proto;
}

TEST(SchemaProtoTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema,
      Schema::FromJson(  //
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
              {"fill_value", 5},
              {"dimension_units", {{4, "nm"}, nullptr, {30, "nm"}}},
          }));

  auto proto = ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
    rank: 3
    dtype: "uint8"
    domain {
      origin: [ 1, 2, 3 ]
      shape: [ 4, 4, 4 ]
      labels: [ "x", "y", "z" ]
    }
    chunk_layout {
      grid_origin: [ -9223372036854775808, -9223372036854775808, 11 ]
      grid_origin_soft_constraint_bitset: 3
      inner_order: [ 2, 0, 1 ]
      inner_order_soft_constraint: true
      write_chunk {
        aspect_ratio: [ 7, 8, 9 ]
        shape: [ 8, 9, 0 ]
        elements: 40
        aspect_ratio_soft_constraint_bitset: 7
        shape_soft_constraint_bitset: 4
      }
      read_chunk {
        shape: [ 6, 0, 7 ]
        elements: 30
        aspect_ratio: [ 4, 5, 6 ]
        shape_soft_constraint_bitset: 7
      }
      codec_chunk {
        elements: 20
        shape: [ 0, 4, 5 ]
        aspect_ratio: [ 1, 2, 3 ]
        elements_soft_constraint: true
        shape_soft_constraint_bitset: 1
      }
    }
    fill_value { dtype: "uint8" void_data: "\x05" }
    dimension_unit { multiplier: 4 base_unit: "nm" }
    dimension_unit {}
    dimension_unit { multiplier: 30 base_unit: "nm" }
  )pb");

  EXPECT_THAT(DoEncode(schema), EqualsProto(proto));
  EXPECT_THAT(ParseSchemaFromProto(proto), testing::Eq(schema));
}

TEST(SchemaProtoTest, Empty) {
  tensorstore::Schema schema;
  EXPECT_THAT(
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
      )pb")),
      testing::Eq(schema));
}

TEST(SchemaProtoTest, RankFromDimensionUnit) {
  // TODO: Fix json output of this case.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema,
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
        rank: 1
        dimension_unit {}
      )pb")));

  EXPECT_THAT(
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
        dimension_unit {}
      )pb")),
      testing::Eq(schema));
}

TEST(SchemaProtoTest, Errors) {
  EXPECT_THAT(
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
        rank: -2
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
        dtype: "foo"
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseSchemaFromProto(ParseProtoOrDie<::tensorstore::proto::Schema>(R"pb(
        codec: "12345"
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
