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
#include "tensorstore/codec_spec.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::internal_zarr::ChunkKeyEncoding;
using tensorstore::internal_zarr::ChunkKeyEncodingJsonBinder;
using tensorstore::internal_zarr::GetCompatibleField;
using tensorstore::internal_zarr::ParseDType;
using tensorstore::internal_zarr::ParseSelectedField;
using tensorstore::internal_zarr::SelectedField;
using tensorstore::internal_zarr::ZarrMetadata;
using tensorstore::internal_zarr::ZarrPartialMetadata;

TEST(ParsePartialMetadataTest, ExtraMember) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"foo", "x"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Object includes extra members: \"foo\"")},
  });
}

TEST(ParsePartialMetadataTest, InvalidZarrFormat) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"zarr_format", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"zarr_format\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidChunks) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"chunks", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"chunks\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidShape) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"shape", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"shape\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidCompressor) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"compressor", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"compressor\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidOrder) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"order", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"order\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidDType) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"dtype", "2"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"dtype\": .*")},
  });
}

TEST(ParsePartialMetadataTest, InvalidFilters) {
  tensorstore::TestJsonBinderFromJson<ZarrPartialMetadata>({
      {{{"filters", "x"}},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing object member \"filters\": .*")},
  });
}

TEST(ParsePartialMetadataTest, Empty) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, ZarrPartialMetadata::FromJson(::nlohmann::json::object_t{}));
  EXPECT_EQ(std::nullopt, result.zarr_format);
  EXPECT_EQ(std::nullopt, result.order);
  EXPECT_EQ(std::nullopt, result.compressor);
  EXPECT_EQ(std::nullopt, result.filters);
  EXPECT_EQ(std::nullopt, result.dtype);
  EXPECT_EQ(std::nullopt, result.fill_value);
  EXPECT_EQ(std::nullopt, result.shape);
  EXPECT_EQ(std::nullopt, result.chunks);
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
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, ZarrPartialMetadata::FromJson(GetMetadataSpec()));
  EXPECT_EQ(2, result.zarr_format);
  EXPECT_EQ(tensorstore::c_order, result.order);
  ASSERT_TRUE(result.compressor);
  EXPECT_EQ((::nlohmann::json{{"id", "blosc"},
                              {"blocksize", 0},
                              {"clevel", 5},
                              {"cname", "lz4"},
                              {"shuffle", -1}}),
            ::nlohmann::json(*result.compressor));

  ASSERT_TRUE(result.dtype);
  EXPECT_EQ("<i2", ::nlohmann::json(*result.dtype));
  ASSERT_TRUE(result.fill_value);
  ASSERT_EQ(1, result.fill_value->size());
  EXPECT_FALSE((*result.fill_value)[0].valid());
  ASSERT_TRUE(result.shape);
  EXPECT_THAT(*result.shape, ::testing::ElementsAre(100, 100));
  ASSERT_TRUE(result.chunks);
  EXPECT_THAT(*result.chunks, ::testing::ElementsAre(3, 2));
}

TEST(ChunkKeyEncodingTest, JsonBinderTest) {
  tensorstore::TestJsonBinderRoundTrip<ChunkKeyEncoding>(
      {
          {ChunkKeyEncoding::kDotSeparated, "."},
          {ChunkKeyEncoding::kSlashSeparated, "/"},
      },
      ChunkKeyEncodingJsonBinder);
}

TEST(ChunkKeyEncodingTest, JsonBinderTestInvalid) {
  tensorstore::TestJsonBinderFromJson<ChunkKeyEncoding>(
      {
          {"x", MatchesStatus(absl::StatusCode::kInvalidArgument)},
          {3, MatchesStatus(absl::StatusCode::kInvalidArgument)},
      },
      ChunkKeyEncodingJsonBinder);
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
                                   tensorstore::dtype_v<std::int32_t>,
                                   /*selected_field=*/SelectedField()));
  EXPECT_THAT(
      GetCompatibleField(ParseDType("<i4").value(),
                         /*data_type_constraint=*/
                         tensorstore::dtype_v<std::uint32_t>,
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
  EXPECT_THAT(
      GetNewMetadata(
          ZarrPartialMetadata::FromJson(
              {{"chunks", {2, 3}}, {"dtype", "<i4"}, {"compressor", nullptr}})
              .value(),
          /*data_type_constraint=*/{}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"shape\" must be specified"));
}

TEST(GetNewMetadataTest, NoChunks) {
  EXPECT_THAT(
      GetNewMetadata(
          ZarrPartialMetadata::FromJson(
              {{"shape", {2, 3}}, {"dtype", "<i4"}, {"compressor", nullptr}})
              .value(),
          /*data_type_constraint=*/{}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"chunks\" must be specified"));
}

TEST(GetNewMetadataTest, NoDtype) {
  EXPECT_THAT(
      GetNewMetadata(
          ZarrPartialMetadata::FromJson(
              {{"shape", {2, 3}}, {"chunks", {2, 3}}, {"compressor", nullptr}})
              .value(),
          /*data_type_constraint=*/{}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"dtype\" must be specified"));
}

TEST(GetNewMetadataTest, NoCompressor) {
  EXPECT_THAT(GetNewMetadata(
                  ZarrPartialMetadata::FromJson(
                      {{"shape", {2, 3}}, {"chunks", {2, 3}}, {"dtype", "<i4"}})
                      .value(),
                  /*data_type_constraint=*/{}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "\"compressor\" must be specified"));
}

TEST(GetNewMetadataTest, IntegerOverflow) {
  EXPECT_THAT(
      GetNewMetadata(
          ZarrPartialMetadata::FromJson(
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
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto partial_metadata, ZarrPartialMetadata::FromJson(GetMetadataSpec()));
  TENSORSTORE_EXPECT_OK(ValidateMetadata(metadata, partial_metadata));
}

TEST(ValidateMetadataTest, Unconstrained) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto partial_metadata,
      ZarrPartialMetadata::FromJson(::nlohmann::json::object_t{}));
  TENSORSTORE_EXPECT_OK(ValidateMetadata(metadata, partial_metadata));
}

TEST(ValidateMetadataTest, ShapeMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["shape"] = {7, 8};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(
      ValidateMetadata(metadata, partial_metadata),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Expected \"shape\" of \\[7,8\\] but received: \\[100,100\\]"));
}

TEST(ValidateMetadataTest, ChunksMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["chunks"] = {1, 1};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(ValidateMetadata(metadata, partial_metadata),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "Expected \"chunks\" of \\[1,1\\] but received: \\[3,2\\]"));
}

TEST(ValidateMetadataTest, OrderMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["order"] = "F";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(ValidateMetadata(metadata, partial_metadata),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"order\" of \"F\" but received: \"C\""));
}

TEST(ValidateMetadataTest, CompressorMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["compressor"] = nullptr;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(ValidateMetadata(metadata, partial_metadata),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"compressor\" of null but received: "
                            "\\{\"blocksize\":0,\"clevel\":5,\"cname\":\"lz4\","
                            "\"id\":\"blosc\",\"shuffle\":-1\\}"));
}

TEST(ValidateMetadataTest, DTypeMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["dtype"] = ">i4";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(
      ValidateMetadata(metadata, partial_metadata),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Expected \"dtype\" of \">i4\" but received: \"<i2\""));
}

TEST(ValidateMetadataTest, FillValueMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ZarrMetadata::FromJson(GetMetadataSpec()));
  ::nlohmann::json spec = GetMetadataSpec();
  spec["fill_value"] = 1;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto partial_metadata,
                                   ZarrPartialMetadata::FromJson(spec));
  EXPECT_THAT(ValidateMetadata(metadata, partial_metadata),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Expected \"fill_value\" of 1 but received: null"));
}

TEST(ZarrEncodingSpecTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<tensorstore::CodecSpec::Ptr>({
      ::nlohmann::json::value_t::discarded,
      {
          {"driver", "zarr"},
          {"compressor", nullptr},
          {"filters", nullptr},
      },
      {
          {"driver", "zarr"},
          {"compressor",
           {{"id", "blosc"},
            {"cname", "lz4"},
            {"clevel", 5},
            {"blocksize", 0},
            {"shuffle", -1}}},
          {"filters", nullptr},
      },
  });
}

}  // namespace
