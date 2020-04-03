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

#include "tensorstore/driver/zarr/spec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::ChunkKeyEncoding;
using tensorstore::internal_zarr::GetCompatibleField;
using tensorstore::internal_zarr::ParseDType;
using tensorstore::internal_zarr::ParseKeyEncoding;
using tensorstore::internal_zarr::ParseMetadata;
using tensorstore::internal_zarr::ParsePartialMetadata;
using tensorstore::internal_zarr::ParseSelectedField;
using tensorstore::internal_zarr::SelectedField;
using tensorstore::internal_zarr::ZarrMetadata;

TEST(ParsePartialMetadataTest, ExtraMember) {
  EXPECT_THAT(ParsePartialMetadata({{"foo", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"foo\""));
}

TEST(ParsePartialMetadataTest, InvalidZarrFormat) {
  EXPECT_THAT(ParsePartialMetadata({{"zarr_format", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"zarr_format\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidChunks) {
  EXPECT_THAT(ParsePartialMetadata({{"chunks", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"chunks\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidShape) {
  EXPECT_THAT(ParsePartialMetadata({{"shape", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"shape\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidCompressor) {
  EXPECT_THAT(ParsePartialMetadata({{"compressor", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"compressor\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidOrder) {
  EXPECT_THAT(ParsePartialMetadata({{"order", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"order\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidDType) {
  EXPECT_THAT(ParsePartialMetadata({{"dtype", "2"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"dtype\": .*"));
}

TEST(ParsePartialMetadataTest, InvalidFilters) {
  EXPECT_THAT(ParsePartialMetadata({{"filters", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"filters\": .*"));
}

TEST(ParsePartialMetadataTest, Empty) {
  auto result = ParsePartialMetadata(::nlohmann::json::object_t{});
  ASSERT_EQ(Status(), GetStatus(result));
  EXPECT_EQ(absl::nullopt, result->zarr_format);
  EXPECT_EQ(absl::nullopt, result->order);
  EXPECT_EQ(absl::nullopt, result->compressor);
  EXPECT_EQ(absl::nullopt, result->dtype);
  EXPECT_EQ(absl::nullopt, result->fill_value);
  EXPECT_EQ(absl::nullopt, result->shape);
  EXPECT_EQ(absl::nullopt, result->chunks);
}

::nlohmann::json GetMetadataSpec() {
  return {{"zarr_format", 2},
          {"chunks", {3, 2}},
          {"shape", {100, 100}},
          {"order", "C"},
          {"filters", nullptr},
          {"fill_value", nullptr},
          {"dtype", "<i2"},
          {"compressor",
           {{"id", "blosc"},
            {"blocksize", 0},
            {"clevel", 5},
            {"cname", "lz4"},
            {"shuffle", -1}}}};
}

TEST(ParsePartialMetadataTest, Complete) {
  auto result = ParsePartialMetadata(GetMetadataSpec());
  ASSERT_EQ(Status(), GetStatus(result));
  EXPECT_EQ(2, result->zarr_format);
  EXPECT_EQ(tensorstore::c_order, result->order);
  ASSERT_TRUE(result->compressor);
  EXPECT_EQ((::nlohmann::json{{"id", "blosc"},
                              {"blocksize", 0},
                              {"clevel", 5},
                              {"cname", "lz4"},
                              {"shuffle", -1}}),
            ::nlohmann::json(*result->compressor));

  ASSERT_TRUE(result->dtype);
  EXPECT_EQ("<i2", ::nlohmann::json(*result->dtype));
  EXPECT_EQ(nullptr, result->fill_value);

  ASSERT_TRUE(result->shape);
  EXPECT_THAT(*result->shape, ::testing::ElementsAre(100, 100));
  ASSERT_TRUE(result->chunks);
  EXPECT_THAT(*result->chunks, ::testing::ElementsAre(3, 2));
}

TEST(ParseKeyEncodingTest, Basic) {
  EXPECT_EQ(ChunkKeyEncoding::kDotSeparated, ParseKeyEncoding("."));
  EXPECT_EQ(ChunkKeyEncoding::kSlashSeparated, ParseKeyEncoding("/"));
}

TEST(KeyEncodingToJsonTest, Basic) {
  EXPECT_EQ(".", ::nlohmann::json(ChunkKeyEncoding::kDotSeparated));
  EXPECT_EQ("/", ::nlohmann::json(ChunkKeyEncoding::kSlashSeparated));
}

TEST(ParseKeyEncodingTest, Invalid) {
  EXPECT_THAT(ParseKeyEncoding("x"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected \"\\.\" or \"/\", but received: \"x\""));
  EXPECT_THAT(ParseKeyEncoding(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected \"\\.\" or \"/\", but received: 3"));
}

TEST(ParseSelectedFieldTest, Null) {
  EXPECT_EQ(SelectedField(), ParseSelectedField(nullptr));
}

TEST(ParseSelectedFieldTest, InvalidString) {
  EXPECT_THAT(
      ParseSelectedField(""),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected null or non-empty string, but received: \"\""));
}

TEST(ParseSelectedFieldTest, String) {
  EXPECT_EQ(SelectedField("label"), ParseSelectedField("label"));
}

TEST(ParseSelectedFieldTest, InvalidType) {
  EXPECT_THAT(
      ParseSelectedField(true),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected null or non-empty string, but received: true"));
}

TEST(GetCompatibleFieldTest, Null) {
  EXPECT_EQ(0u, GetCompatibleField(ParseDType("<i4").value(),
                                   /*data_type_constraint=*/{},
                                   /*selected_field=*/SelectedField()));
  EXPECT_EQ(0u, GetCompatibleField(ParseDType("<i4").value(),
                                   /*data_type_constraint=*/
                                   tensorstore::DataTypeOf<std::int32_t>(),
                                   /*selected_field=*/SelectedField()));
  EXPECT_THAT(
      GetCompatibleField(ParseDType("<i4").value(),
                         /*data_type_constraint=*/
                         tensorstore::DataTypeOf<std::uint32_t>(),
                         /*selected_field=*/SelectedField()),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Expected field to have data type of uint32 but the actual "
                    "data type is: int32"));
  EXPECT_THAT(
      GetCompatibleField(
          ParseDType(::nlohmann::json::array_t{{"x", "<i4"}, {"y", "<u2"}})
              .value(),
          /*data_type_constraint=*/{},
          /*selected_field=*/SelectedField()),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Must specify a \"field\" that is one of: \\[\"x\",\"y\"\\]"));
}

TEST(GetCompatibleFieldTest, String) {
  EXPECT_THAT(
      GetCompatibleField(ParseDType("<i4").value(),
                         /*data_type_constraint=*/{},
                         /*selected_field=*/"x"),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Requested field \"x\" but dtype does not have named fields"));
  EXPECT_EQ(0u, GetCompatibleField(ParseDType(::nlohmann::json::array_t{
                                                  {"x", "<i4"}, {"y", "<u2"}})
                                       .value(),
                                   /*data_type_constraint=*/{},
                                   /*selected_field=*/"x"));
  EXPECT_EQ(1u, GetCompatibleField(ParseDType(::nlohmann::json::array_t{
                                                  {"x", "<i4"}, {"y", "<u2"}})
                                       .value(),
                                   /*data_type_constraint=*/{},
                                   /*selected_field=*/"y"));

  EXPECT_THAT(
      GetCompatibleField(
          ParseDType(::nlohmann::json::array_t{{"x", "<i4"}, {"y", "<u2"}})
              .value(),
          /*data_type_constraint=*/{},
          /*selected_field=*/"z"),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Requested field \"z\" is not one of: \\[\"x\",\"y\"\\]"));
}

TEST(EncodeSelectedFieldTest, NonEmpty) {
  auto dtype =
      ParseDType(::nlohmann::json::array_t{{"x", "<i4"}, {"y", "<u2"}}).value();
  EXPECT_EQ("x", EncodeSelectedField(0, dtype));
  EXPECT_EQ("y", EncodeSelectedField(1, dtype));
}

TEST(EncodeSelectedFieldTest, Empty) {
  auto dtype = ParseDType("<i4").value();
  // dtype does not have multiple fields.  `EncodeSelectedField` returns the
  // empty string to indicate that.
  EXPECT_EQ("", EncodeSelectedField(0, dtype));
}

TEST(GetNewMetadataTest, NoShape) {
  EXPECT_THAT(GetNewMetadata(ParsePartialMetadata({{"chunks", {2, 3}},
                                                   {"dtype", "<i4"},
                                                   {"compressor", nullptr}})
                                 .value(),
                             /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "\"shape\" must be specified"));
}

TEST(GetNewMetadataTest, NoChunks) {
  EXPECT_THAT(GetNewMetadata(ParsePartialMetadata({{"shape", {2, 3}},
                                                   {"dtype", "<i4"},
                                                   {"compressor", nullptr}})
                                 .value(),
                             /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "\"chunks\" must be specified"));
}

TEST(GetNewMetadataTest, NoDtype) {
  EXPECT_THAT(GetNewMetadata(ParsePartialMetadata({{"shape", {2, 3}},
                                                   {"chunks", {2, 3}},
                                                   {"compressor", nullptr}})
                                 .value(),
                             /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "\"dtype\" must be specified"));
}

TEST(GetNewMetadataTest, NoCompressor) {
  EXPECT_THAT(GetNewMetadata(
                  ParsePartialMetadata(
                      {{"shape", {2, 3}}, {"chunks", {2, 3}}, {"dtype", "<i4"}})
                      .value(),
                  /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "\"compressor\" must be specified"));
}

TEST(GetNewMetadataTest, InvalidFillValue) {
  EXPECT_THAT(GetNewMetadata(ParsePartialMetadata({{"shape", {2, 3}},
                                                   {"chunks", {2, 3}},
                                                   {"dtype", "<i4"},
                                                   {"compressor", nullptr},
                                                   {"fill_value", "x"}})
                                 .value(),
                             /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"fill_value\": "
                            ".*"));
}

TEST(GetNewMetadataTest, IntegerOverflow) {
  EXPECT_THAT(
      GetNewMetadata(
          ParsePartialMetadata(
              {{"shape", {4611686018427387903, 4611686018427387903}},
               {"chunks", {4611686018427387903, 4611686018427387903}},
               {"dtype", "<i4"},
               {"compressor", nullptr}})
              .value(),
          /*data_type_constraint=*/{}),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Product of chunk dimensions "
          "\\{4611686018427387903, 4611686018427387903\\} is too large"));
}

TEST(ValidateMetadataTest, Success) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  EXPECT_EQ(Status(),
            ValidateMetadata(metadata,
                             ParsePartialMetadata(GetMetadataSpec()).value()));
}

TEST(ValidateMetadataTest, Unconstrained) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  EXPECT_EQ(Status(),
            ValidateMetadata(
                metadata,
                ParsePartialMetadata(::nlohmann::json::object_t{}).value()));
}

TEST(ValidateMetadataTest, ZarrFormatMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  // Currently only a `zarr_format` of 2 is supported, so we have to artifically
  // change it here to test the validation.
  metadata.zarr_format = 3;
  EXPECT_THAT(ValidateMetadata(metadata,
                               ParsePartialMetadata(GetMetadataSpec()).value()),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"zarr_format\" of 2 but received: 3"));
}

TEST(ValidateMetadataTest, ShapeMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["shape"] = {7, 8};
  EXPECT_THAT(
      ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Expected \"shape\" of \\[7,8\\] but received: \\[100,100\\]"));
}

TEST(ValidateMetadataTest, ChunksMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["chunks"] = {1, 1};
  EXPECT_THAT(ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Expected \"chunks\" of \\[1,1\\] but received: \\[3,2\\]"));
}

TEST(ValidateMetadataTest, OrderMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["order"] = "F";
  EXPECT_THAT(ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"order\" of \"F\" but received: \"C\""));
}

TEST(ValidateMetadataTest, CompressorMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["compressor"] = nullptr;
  EXPECT_THAT(ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"compressor\" of null but received: "
                            "\\{\"blocksize\":0,\"clevel\":5,\"cname\":\"lz4\","
                            "\"id\":\"blosc\",\"shuffle\":-1\\}"));
}

TEST(ValidateMetadataTest, DTypeMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["dtype"] = ">i4";
  EXPECT_THAT(
      ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Expected \"dtype\" of \">i4\" but received: \"<i2\""));
}

TEST(ValidateMetadataTest, FillValueMismatch) {
  ZarrMetadata metadata;
  ASSERT_TRUE(ParseMetadata(GetMetadataSpec(), &metadata).ok());
  ::nlohmann::json spec = GetMetadataSpec();
  spec["fill_value"] = 1;
  EXPECT_THAT(ValidateMetadata(metadata, ParsePartialMetadata(spec).value()),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"fill_value\" of 1 but received: null"));
}

}  // namespace
