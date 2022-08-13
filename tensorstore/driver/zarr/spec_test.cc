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
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::ChunkLayout;
using ::tensorstore::CodecSpec;
using ::tensorstore::dtype_v;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Schema;
using ::tensorstore::internal_zarr::GetFieldIndex;
using ::tensorstore::internal_zarr::ParseDType;
using ::tensorstore::internal_zarr::ParseSelectedField;
using ::tensorstore::internal_zarr::SelectedField;
using ::tensorstore::internal_zarr::ZarrMetadata;
using ::tensorstore::internal_zarr::ZarrPartialMetadata;

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

TEST(GetFieldIndexTest, Null) {
  EXPECT_EQ(0u, GetFieldIndex(ParseDType("<i4").value(), SelectedField()));
  EXPECT_THAT(
      GetFieldIndex(
          ParseDType(::nlohmann::json::array_t{{"x", "<i4"}, {"y", "<u2"}})
              .value(),
          SelectedField()),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Must specify a \"field\" that is one of: \\[\"x\",\"y\"\\]"));
}

TEST(GetFieldIndexTest, String) {
  EXPECT_THAT(
      GetFieldIndex(ParseDType("<i4").value(), "x"),
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          "Requested field \"x\" but dtype does not have named fields"));
  EXPECT_EQ(0u, GetFieldIndex(ParseDType(::nlohmann::json::array_t{
                                             {"x", "<i4"}, {"y", "<u2"}})
                                  .value(),
                              "x"));
  EXPECT_EQ(1u, GetFieldIndex(ParseDType(::nlohmann::json::array_t{
                                             {"x", "<i4"}, {"y", "<u2"}})
                                  .value(),
                              "y"));

  EXPECT_THAT(
      GetFieldIndex(
          ParseDType(::nlohmann::json::array_t{{"x", "<i4"}, {"y", "<u2"}})
              .value(),
          "z"),
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

template <typename... Option>
tensorstore::Result<::nlohmann::json> GetNewMetadataFromOptions(
    ::nlohmann::json partial_metadata_json, std::string selected_field,
    Option&&... option) {
  Schema schema;
  if (absl::Status status;
      !((status = schema.Set(std::forward<Option>(option))).ok() && ...)) {
    return status;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto partial_metadata,
      ZarrPartialMetadata::FromJson(partial_metadata_json));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_metadata,
      GetNewMetadata(partial_metadata, selected_field, schema));
  return new_metadata->ToJson();
}

TEST(GetNewMetadataTest, FullMetadata) {
  EXPECT_THAT(GetNewMetadataFromOptions({{"chunks", {8, 10}},
                                         {"dtype", "<i4"},
                                         {"compressor", nullptr},
                                         {"shape", {5, 6}}},
                                        /*selected_field=*/{}),
              ::testing::Optional(MatchesJson({
                  {"chunks", {8, 10}},
                  {"compressor", nullptr},
                  {"dtype", "<i4"},
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"order", "C"},
                  {"shape", {5, 6}},
                  {"zarr_format", 2},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, NoShape) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"chunks", {2, 3}}, {"dtype", "<i4"}, {"compressor", nullptr}},
          /*selected_field=*/{}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "domain must be specified"));
}

TEST(GetNewMetadataTest, AutomaticChunks) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"shape", {2, 3}}, {"dtype", "<i4"}, {"compressor", nullptr}},
          /*selected_field=*/{}),
      ::testing::Optional(MatchesJson({
          {"chunks", {2, 3}},
          {"compressor", nullptr},
          {"dtype", "<i4"},
          {"fill_value", nullptr},
          {"filters", nullptr},
          {"order", "C"},
          {"shape", {2, 3}},
          {"zarr_format", 2},
          {"dimension_separator", "."},
      })));
}

TEST(GetNewMetadataTest, NoDtype) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"shape", {2, 3}}, {"chunks", {2, 3}}, {"compressor", nullptr}},
          /*selected_field=*/{}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"dtype\" must be specified"));
}

TEST(GetNewMetadataTest, NoCompressor) {
  EXPECT_THAT(GetNewMetadataFromOptions(
                  {{"shape", {2, 3}}, {"chunks", {2, 3}}, {"dtype", "<i4"}},
                  /*selected_field=*/{}),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {2, 3}},
                  {"chunks", {2, 3}},
                  {"dtype", "<i4"},
                  {"compressor",
                   {
                       {"id", "blosc"},
                       {"cname", "lz4"},
                       {"clevel", 5},
                       {"blocksize", 0},
                       {"shuffle", -1},
                   }},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, IntegerOverflow) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"shape", {4611686018427387903, 4611686018427387903}},
           {"chunks", {4611686018427387903, 4611686018427387903}},
           {"dtype", "<i4"},
           {"compressor", nullptr}},
          /*selected_field=*/{}),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Product of chunk dimensions "
          "\\{4611686018427387903, 4611686018427387903\\} is too large"));
}

TEST(GetNewMetadataTest, SchemaDomainDtype) {
  EXPECT_THAT(GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                        /*selected_field=*/{},
                                        tensorstore::IndexDomainBuilder(3)
                                            .shape({1000, 2000, 3000})
                                            .Finalize()
                                            .value(),
                                        dtype_v<int32_t>),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {1000, 2000, 3000}},
                  {"chunks", {102, 102, 102}},
                  {"dtype", "<i4"},
                  {"compressor",
                   {
                       {"id", "blosc"},
                       {"cname", "lz4"},
                       {"clevel", 5},
                       {"blocksize", 0},
                       {"shuffle", -1},
                   }},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDomainDtypeFillValue) {
  EXPECT_THAT(GetNewMetadataFromOptions(
                  ::nlohmann::json::object_t(),
                  /*selected_field=*/{},
                  tensorstore::IndexDomainBuilder(3)
                      .shape({1000, 2000, 3000})
                      .Finalize()
                      .value(),
                  dtype_v<int32_t>,
                  Schema::FillValue{tensorstore::MakeScalarArray<int32_t>(5)}),
              ::testing::Optional(MatchesJson({
                  {"fill_value", 5},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {1000, 2000, 3000}},
                  {"chunks", {102, 102, 102}},
                  {"dtype", "<i4"},
                  {"compressor",
                   {
                       {"id", "blosc"},
                       {"cname", "lz4"},
                       {"clevel", 5},
                       {"blocksize", 0},
                       {"shuffle", -1},
                   }},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaObjectWithDomainDtypeFillValue) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::IndexDomainBuilder(3)
                                       .shape({1000, 2000, 3000})
                                       .Finalize()
                                       .value()));
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<int32_t>));
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::FillValue{tensorstore::MakeScalarArray<int32_t>(5)}));
  EXPECT_THAT(GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                        /*selected_field=*/{}, schema),
              ::testing::Optional(MatchesJson({
                  {"fill_value", 5},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {1000, 2000, 3000}},
                  {"chunks", {102, 102, 102}},
                  {"dtype", "<i4"},
                  {"compressor",
                   {
                       {"id", "blosc"},
                       {"cname", "lz4"},
                       {"clevel", 5},
                       {"blocksize", 0},
                       {"shuffle", -1},
                   }},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDtypeShapeCodec) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec,
      CodecSpec::FromJson({{"driver", "zarr"}, {"compressor", nullptr}}));
  EXPECT_THAT(GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                        /*selected_field=*/{},
                                        Schema::Shape({100, 200}),
                                        dtype_v<int32_t>, codec),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {100, 200}},
                  {"chunks", {100, 200}},
                  {"dtype", "<i4"},
                  {"compressor", nullptr},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDtypeInnerOrderC) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec,
      CodecSpec::FromJson({{"driver", "zarr"}, {"compressor", nullptr}}));
  EXPECT_THAT(GetNewMetadataFromOptions(
                  ::nlohmann::json::object_t(),
                  /*selected_field=*/{}, Schema::Shape({100, 200}),
                  ChunkLayout::InnerOrder({0, 1}), dtype_v<int32_t>, codec),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {100, 200}},
                  {"chunks", {100, 200}},
                  {"dtype", "<i4"},
                  {"compressor", nullptr},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDtypeInnerOrderFortran) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec,
      CodecSpec::FromJson({{"driver", "zarr"}, {"compressor", nullptr}}));
  EXPECT_THAT(GetNewMetadataFromOptions(
                  ::nlohmann::json::object_t(),
                  /*selected_field=*/{}, Schema::Shape({100, 200}),
                  ChunkLayout::InnerOrder({1, 0}), dtype_v<int32_t>, codec),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "F"},
                  {"shape", {100, 200}},
                  {"chunks", {100, 200}},
                  {"dtype", "<i4"},
                  {"compressor", nullptr},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDtypeInnerOrderFortranFieldShape) {
  EXPECT_THAT(GetNewMetadataFromOptions(
                  {
                      {"compressor", nullptr},
                      {"dtype", {{"x", "<u4", {2, 3}}}},
                  },
                  /*selected_field=*/"x", Schema::Shape({100, 200, 2, 3}),
                  ChunkLayout::InnerOrder({1, 0, 2, 3})),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "F"},
                  {"shape", {100, 200}},
                  {"chunks", {100, 200}},
                  {"dtype", {{"x", "<u4", {2, 3}}}},
                  {"compressor", nullptr},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaDtypeInnerOrderInvalid) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          ::nlohmann::json::object_t(),
          /*selected_field=*/{}, Schema::Shape({100, 200, 300}),
          ChunkLayout::InnerOrder({2, 0, 1}), dtype_v<int32_t>),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid \"inner_order\" constraint: \\{2, 0, 1\\}"));
}

TEST(GetNewMetadataTest, SchemaDtypeInnerOrderInvalidSoft) {
  EXPECT_THAT(GetNewMetadataFromOptions(
                  {{"compressor", nullptr}},
                  /*selected_field=*/{}, Schema::Shape({100, 200, 300}),
                  ChunkLayout::InnerOrder({2, 0, 1}, /*hard_constraint=*/false),
                  dtype_v<int32_t>),
              ::testing::Optional(MatchesJson({
                  {"fill_value", nullptr},
                  {"filters", nullptr},
                  {"zarr_format", 2},
                  {"order", "C"},
                  {"shape", {100, 200, 300}},
                  {"chunks", {100, 102, 102}},
                  {"dtype", "<i4"},
                  {"compressor", nullptr},
                  {"dimension_separator", "."},
              })));
}

TEST(GetNewMetadataTest, SchemaStructuredDtypeInvalidFillValue) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"dtype", ::nlohmann::json::array_t{{"x", "<u4"}, {"y", "<i4"}}}},
          /*selected_field=*/"x", Schema::Shape({100, 200}),
          Schema::FillValue(tensorstore::MakeScalarArray<uint32_t>(42))),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Invalid fill_value: Cannot specify fill_value through schema for "
          "structured zarr data type \\[.*"));
}

TEST(GetNewMetadataTest, SchemaFillValueMismatch) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"dtype", "<u4"}, {"fill_value", 42}},
          /*selected_field=*/{}, Schema::Shape({100, 200}),
          Schema::FillValue(tensorstore::MakeScalarArray<uint32_t>(43))),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid fill_value: .*"));
}

TEST(GetNewMetadataTest, SchemaFillValueMismatchNull) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {{"dtype", "<u4"}, {"fill_value", nullptr}},
          /*selected_field=*/{}, Schema::Shape({100, 200}),
          Schema::FillValue(tensorstore::MakeScalarArray<uint32_t>(42))),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid fill_value: .*"));
}

TEST(GetNewMetadataTest, SchemaFillValueRedundant) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(
          {
              {"dtype", "<u4"},
              {"fill_value", 42},
              {"compressor", nullptr},
          },
          /*selected_field=*/{}, Schema::Shape({100, 200}),
          Schema::FillValue(tensorstore::MakeScalarArray<uint32_t>(42))),
      ::testing::Optional(MatchesJson({
          {"fill_value", 42},
          {"filters", nullptr},
          {"zarr_format", 2},
          {"order", "C"},
          {"shape", {100, 200}},
          {"chunks", {100, 200}},
          {"dtype", "<u4"},
          {"compressor", nullptr},
          {"dimension_separator", "."},
      })));
}

TEST(GetNewMetadataTest, SchemaCodecChunkShape) {
  EXPECT_THAT(GetNewMetadataFromOptions(
                  ::nlohmann::json::object_t{},
                  /*selected_field=*/{}, Schema::Shape({100, 200}),
                  dtype_v<uint32_t>, ChunkLayout::CodecChunkShape({5, 6})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "codec_chunk_shape not supported"));
}

TEST(GetNewMetadataTest, CodecMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec,
      CodecSpec::FromJson({{"driver", "zarr"}, {"compressor", nullptr}}));
  EXPECT_THAT(
      GetNewMetadataFromOptions({{"compressor", {{"id", "blosc"}}}},
                                /*selected_field=*/{},
                                Schema::Shape({100, 200}), dtype_v<int32_t>,
                                codec),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot merge codec spec .* with .*: \"compressor\" does not match"));
}

TEST(GetNewMetadataTest, SelectedFieldDtypeNotSpecified) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                /*selected_field=*/"x",
                                Schema::Shape({100, 200}), dtype_v<int32_t>),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "\"dtype\" must be specified in \"metadata\" if "
                    "\"field\" is specified"));
}

TEST(GetNewMetadataTest, SelectedFieldInvalid) {
  EXPECT_THAT(
      GetNewMetadataFromOptions({{"dtype", {{"x", "<u4", {2}}, {"y", "<i4"}}}},
                                /*selected_field=*/"z",
                                Schema::Shape({100, 200})),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Requested field \"z\" is not one of: \\[\"x\",\"y\"\\]"));
}

TEST(GetNewMetadataTest, InvalidDtype) {
  EXPECT_THAT(GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                        /*selected_field=*/{},
                                        dtype_v<tensorstore::json_t>,
                                        Schema::Shape({100, 200})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Data type not supported: json"));
}

TEST(GetNewMetadataTest, InvalidDomain) {
  EXPECT_THAT(
      GetNewMetadataFromOptions(::nlohmann::json::object_t(),
                                /*selected_field=*/{},
                                dtype_v<tensorstore::int32_t>,
                                tensorstore::IndexDomainBuilder(2)
                                    .origin({1, 2})
                                    .shape({100, 200})
                                    .Finalize()
                                    .value()),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid domain: .*"));
}

TEST(GetNewMetadataTest, DomainIncompatibleWithFieldShape) {
  EXPECT_THAT(
      GetNewMetadataFromOptions({{"dtype", {{"x", "<u4", {2, 3}}}}},
                                /*selected_field=*/"x",
                                Schema::Shape({100, 200, 2, 4})),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Invalid domain: .*"));
}

TEST(GetNewMetadataTest, DomainIncompatibleWithMetadataRank) {
  EXPECT_THAT(
      GetNewMetadataFromOptions({{"chunks", {100, 100}}},
                                /*selected_field=*/{},
                                dtype_v<tensorstore::int32_t>,
                                Schema::Shape({100, 200, 300})),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank specified by schema \\(3\\) is not compatible with metadata"));
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

TEST(ZarrCodecSpecTest, Merge) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec1,
                                   CodecSpec::FromJson({{"driver", "zarr"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec2,
      CodecSpec::FromJson({{"driver", "zarr"}, {"filters", nullptr}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec3,
      CodecSpec::FromJson({{"driver", "zarr"}, {"compressor", nullptr}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec4, CodecSpec::FromJson({{"driver", "zarr"},
                                        {"compressor", {{"id", "blosc"}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec5,
      CodecSpec::FromJson(
          {{"driver", "zarr"}, {"compressor", nullptr}, {"filters", nullptr}}));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec1), ::testing::Optional(codec1));
  EXPECT_THAT(CodecSpec::Merge(codec3, codec3), ::testing::Optional(codec3));
  EXPECT_THAT(CodecSpec::Merge(codec1, CodecSpec()),
              ::testing::Optional(codec1));
  EXPECT_THAT(CodecSpec::Merge(CodecSpec(), codec1),
              ::testing::Optional(codec1));
  EXPECT_THAT(CodecSpec::Merge(CodecSpec(), CodecSpec()),
              ::testing::Optional(CodecSpec()));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec2), ::testing::Optional(codec2));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec3), ::testing::Optional(codec3));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec3), ::testing::Optional(codec5));
  EXPECT_THAT(
      CodecSpec::Merge(codec3, codec4),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot merge codec spec .* with .*: \"compressor\" does not match"));
}

TEST(ZarrCodecSpecTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<tensorstore::CodecSpec>({
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
