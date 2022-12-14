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

#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {
using ::tensorstore::Box;
using ::tensorstore::CodecSpec;
using ::tensorstore::DataType;
using ::tensorstore::DataTypeId;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::kDataTypes;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::StrCat;
using ::tensorstore::internal::ParseJson;
using ::tensorstore::internal_neuroglancer_precomputed::EncodeCompressedZIndex;
using ::tensorstore::internal_neuroglancer_precomputed::
    GetChunksPerVolumeShardFunction;
using ::tensorstore::internal_neuroglancer_precomputed::GetCompressedZIndexBits;
using ::tensorstore::internal_neuroglancer_precomputed::
    GetMetadataCompatibilityKey;
using ::tensorstore::internal_neuroglancer_precomputed::GetShardChunkHierarchy;
using ::tensorstore::internal_neuroglancer_precomputed::MultiscaleMetadata;
using ::tensorstore::internal_neuroglancer_precomputed::
    MultiscaleMetadataConstraints;
using ::tensorstore::internal_neuroglancer_precomputed::NoShardingSpec;
using ::tensorstore::internal_neuroglancer_precomputed::OpenConstraints;
using ::tensorstore::internal_neuroglancer_precomputed::ResolveScaleKey;
using ::tensorstore::internal_neuroglancer_precomputed::ScaleMetadata;
using ::tensorstore::internal_neuroglancer_precomputed::
    ScaleMetadataConstraints;
using ::tensorstore::internal_neuroglancer_precomputed::ShardChunkHierarchy;
using ::tensorstore::internal_neuroglancer_precomputed::ShardingSpec;
using ::tensorstore::internal_neuroglancer_precomputed::ValidateDataType;
using ::tensorstore::internal_neuroglancer_precomputed::
    ValidateMetadataCompatibility;
using ::testing::ElementsAre;

using Encoding = ScaleMetadata::Encoding;

TEST(NoShardingSpecTest, Basic) {
  EXPECT_TRUE(NoShardingSpec() == NoShardingSpec());
  EXPECT_FALSE(NoShardingSpec() != NoShardingSpec());
  EXPECT_EQ(::nlohmann::json(nullptr), ::nlohmann::json(NoShardingSpec()));
}

TEST(EncodingTest, ToString) {
  EXPECT_EQ("raw", to_string(ScaleMetadata::Encoding::raw));
  EXPECT_EQ("jpeg", to_string(ScaleMetadata::Encoding::jpeg));
  EXPECT_EQ("compressed_segmentation",
            to_string(ScaleMetadata::Encoding::compressed_segmentation));
}

TEST(MetadataTest, ParseUnsharded) {
  ::nlohmann::json metadata_json = ParseJson(R"(
{
  "data_type": "uint8",
  "num_channels": 1,
  "scales": [
    {
      "chunk_sizes": [[64, 64, 64], [128, 2, 64]],
      "encoding": "jpeg",
      "key": "8_8_8",
      "resolution": [8, 8, 8],
      "size": [6446, 6643, 8090],
      "voxel_offset": [-2, 4, 6]
    },
    {
      "chunk_sizes": [[64, 64, 64]],
      "encoding": "jpeg",
      "key": "16_16_16",
      "resolution": [16, 16, 16],
      "size": [3223, 3321, 4045],
      "voxel_offset": [-1, 2, 3]
    }
    ],
  "type": "image"
}
)");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto m,
                                   MultiscaleMetadata::FromJson(metadata_json));
  EXPECT_EQ("image", m.type);
  EXPECT_EQ(dtype_v<std::uint8_t>, m.dtype);
  EXPECT_EQ(1, m.num_channels);
  EXPECT_EQ(::nlohmann::json::object_t{}, m.extra_attributes);
  ASSERT_EQ(2, m.scales.size());
  {
    auto& s = m.scales[0];
    EXPECT_EQ("8_8_8", s.key);
    EXPECT_EQ(Encoding::jpeg, s.encoding);
    EXPECT_THAT(s.resolution, ElementsAre(8, 8, 8));
    EXPECT_THAT(s.chunk_sizes,
                ElementsAre(ElementsAre(64, 64, 64), ElementsAre(128, 2, 64)));
    EXPECT_EQ(Box({-2, 4, 6}, {6446, 6643, 8090}), s.box);
  }
  {
    auto& s = m.scales[1];
    EXPECT_EQ("16_16_16", s.key);
    EXPECT_EQ(Encoding::jpeg, s.encoding);
    EXPECT_THAT(s.resolution, ElementsAre(16, 16, 16));
    EXPECT_THAT(s.chunk_sizes, ElementsAre(ElementsAre(64, 64, 64)));
    EXPECT_EQ(Box({-1, 2, 3}, {3223, 3321, 4045}), s.box);
  }
}

TEST(MetadataTest, ParseSharded) {
  ::nlohmann::json metadata_json = ParseJson(R"(
{
  "@type": "neuroglancer_multiscale_volume",
  "data_type": "uint64",
  "num_channels": 2,
  "scales": [
    {
      "chunk_sizes": [[64, 65, 66]],
      "encoding": "compressed_segmentation",
      "compressed_segmentation_block_size": [8, 9, 10],
      "key": "8_8_8",
      "resolution": [5, 6, 7],
      "size": [6446, 6643, 8090],
      "voxel_offset": [2, 4, 6],
      "sharding":
        {
          "@type": "neuroglancer_uint64_sharded_v1",
          "data_encoding": "gzip",
          "hash": "identity",
          "minishard_bits": 6,
          "minishard_index_encoding": "gzip",
          "preshift_bits": 9,
          "shard_bits": 11
        },
      "extra_scale_attribute": "scale_attribute_value"
    }
    ],
  "type": "segmentation",
  "extra_attribute": "attribute_value"
}
)");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto m,
                                   MultiscaleMetadata::FromJson(metadata_json));
  EXPECT_EQ("segmentation", m.type);
  EXPECT_EQ(dtype_v<std::uint64_t>, m.dtype);
  EXPECT_EQ(2, m.num_channels);
  EXPECT_THAT(m.extra_attributes,
              MatchesJson({{"extra_attribute", "attribute_value"}}));
  ASSERT_EQ(1, m.scales.size());
  {
    auto& s = m.scales[0];
    EXPECT_THAT(
        s.extra_attributes,
        MatchesJson({{"extra_scale_attribute", "scale_attribute_value"}}));
    EXPECT_EQ("8_8_8", s.key);
    EXPECT_EQ(Encoding::compressed_segmentation, s.encoding);
    EXPECT_THAT(s.compressed_segmentation_block_size, ElementsAre(8, 9, 10));
    EXPECT_THAT(s.resolution, ElementsAre(5, 6, 7));
    EXPECT_THAT(s.chunk_sizes, ElementsAre(ElementsAre(64, 65, 66)));
    EXPECT_EQ(Box({2, 4, 6}, {6446, 6643, 8090}), s.box);
    ASSERT_TRUE(std::holds_alternative<ShardingSpec>(s.sharding));
    auto& sharding = std::get<ShardingSpec>(s.sharding);
    EXPECT_EQ(sharding.preshift_bits, 9);
    EXPECT_EQ(sharding.minishard_bits, 6);
    EXPECT_EQ(sharding.shard_bits, 11);
    EXPECT_EQ(sharding.hash_function, ShardingSpec::HashFunction::identity);
    EXPECT_EQ(sharding.data_encoding, ShardingSpec::DataEncoding::gzip);
    EXPECT_EQ(sharding.minishard_index_encoding,
              ShardingSpec::DataEncoding::gzip);
  }

  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["chunk_sizes"] = {{64, 64, 64}, {64, 65, 66}};
    EXPECT_THAT(
        MultiscaleMetadata::FromJson(invalid_json),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            ".*: Sharded format does not support more than one chunk size"));
  }
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["chunk_sizes"] = {{0, 3, 4}};
    EXPECT_THAT(
        MultiscaleMetadata::FromJson(invalid_json),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            ".*: Expected integer in the range \\[1, .*\\], but received: 0"));
  }
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["chunk_sizes"] = ::nlohmann::json::array_t{};
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*: At least one chunk size must be specified"));
  }

  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["encoding"] = "raw";
    EXPECT_THAT(
        MultiscaleMetadata::FromJson(invalid_json),
        MatchesStatus(absl::StatusCode::kInvalidArgument,
                      ".*: Error parsing object member "
                      "\"compressed_segmentation_block_size\": "
                      "Only valid for \"compressed_segmentation\" encoding"));
  }
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0].erase("compressed_segmentation_block_size");
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"compressed_segmentation_block_size\".*"));
  }
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["compressed_segmentation_block_size"] = {0, 2, 3};
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"compressed_segmentation_block_size\".*"));
  }
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["compressed_segmentation_block_size"] =  //
        {0, 2, 3, 4};
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"compressed_segmentation_block_size\".*"));
  }
}

// Tests that `voxel_offset` defaults to `[0, 0, 0]`.
TEST(MetadataTest, ParseDefaultVoxelOffset) {
  ::nlohmann::json metadata_json{{"@type", "neuroglancer_multiscale_volume"},
                                 {"num_channels", 1},
                                 {"scales",
                                  {{{"chunk_sizes", {{64, 65, 66}}},
                                    {"encoding", "raw"},
                                    {"key", "8_8_8"},
                                    {"resolution", {5, 6, 7}},
                                    {"size", {6446, 6643, 8090}}}}},
                                 {"type", "image"},
                                 {"data_type", "uint8"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto m,
                                   MultiscaleMetadata::FromJson(metadata_json));
  EXPECT_EQ(Box({6446, 6643, 8090}), m.scales.at(0).box);
}

TEST(MetadataTest, ParseEncodingsAndDataTypes) {
  const auto GetMetadata = [](::nlohmann::json dtype, ::nlohmann::json encoding,
                              int num_channels = 1,
                              ::nlohmann::json jpeg_quality =
                                  ::nlohmann::json::value_t::discarded) {
    ::nlohmann::json metadata_json{{"num_channels", num_channels},
                                   {"scales",
                                    {{{"chunk_sizes", {{64, 65, 66}}},
                                      {"encoding", encoding},
                                      {"key", "8_8_8"},
                                      {"resolution", {5, 6, 7}},
                                      {"size", {6446, 6643, 8090}},
                                      {"voxel_offset", {2, 4, 6}}}}},
                                   {"type", "segmentation"},
                                   {"data_type", dtype}};
    if (encoding == "compressed_segmentation") {
      metadata_json["scales"][0]["compressed_segmentation_block_size"] =  //
          {8, 8, 8};
    }
    if (!jpeg_quality.is_discarded()) {
      metadata_json["scales"][0]["jpeg_quality"] = jpeg_quality;
    }
    return metadata_json;
  };

  // Test number in place of string data type.
  EXPECT_THAT(MultiscaleMetadata::FromJson(
                  GetMetadata(3, ScaleMetadata::Encoding::raw)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test invalid data type name.
  EXPECT_THAT(MultiscaleMetadata::FromJson(
                  GetMetadata("invalid", ScaleMetadata::Encoding::raw)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test invalid encoding JSON type.
  EXPECT_THAT(MultiscaleMetadata::FromJson(GetMetadata("uint8", 123456)),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*123456.*"));

  // Test invalid encoding name.
  EXPECT_THAT(
      MultiscaleMetadata::FromJson(GetMetadata("uint8", "invalid_encoding")),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"invalid_encoding\".*"));

  // Test valid data types for `raw` encoding.
  for (auto data_type_id :
       {DataTypeId::uint8_t, DataTypeId::uint16_t, DataTypeId::uint32_t,
        DataTypeId::uint64_t, DataTypeId::float32_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto m, MultiscaleMetadata::FromJson(
                    GetMetadata(dtype.name(), ScaleMetadata::Encoding::raw)));
    ASSERT_EQ(1, m.scales.size());
    EXPECT_EQ(dtype, m.dtype);
    EXPECT_EQ(ScaleMetadata::Encoding::raw, m.scales[0].encoding);
  }

  // Test that "jpeg_quality" is not valid for `raw` encoding.
  EXPECT_THAT(
      MultiscaleMetadata::FromJson(
          GetMetadata("uint8", ScaleMetadata::Encoding::raw, 1, 75)),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"jpeg\".*"));

  // Test that "jpeg_quality" is not valid for `compressed_segmentation`
  // encoding.
  EXPECT_THAT(
      MultiscaleMetadata::FromJson(GetMetadata(
          "uint32", ScaleMetadata::Encoding::compressed_segmentation, 1, 75)),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"jpeg\".*"));

  // Test invalid data types for `raw` encoding.
  for (auto data_type_id : {DataTypeId::string_t, DataTypeId::json_t,
                            DataTypeId::ustring_t, DataTypeId::bool_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    EXPECT_THAT(MultiscaleMetadata::FromJson(
                    GetMetadata(dtype.name(), ScaleMetadata::Encoding::raw)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test valid data types and number of channels for `jpeg` encoding.
  for (auto data_type_id : {DataTypeId::uint8_t}) {
    for (int num_channels : {1, 3}) {
      const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto m,
          MultiscaleMetadata::FromJson(GetMetadata(
              dtype.name(), ScaleMetadata::Encoding::jpeg, num_channels)));
      ASSERT_EQ(1, m.scales.size());
      EXPECT_EQ(dtype, m.dtype);
      EXPECT_EQ(num_channels, m.num_channels);
      EXPECT_EQ(75, m.scales[0].jpeg_quality);
      EXPECT_EQ(ScaleMetadata::Encoding::jpeg, m.scales[0].encoding);
    }
  }

  // Test invalid jpeg_quality values.
  EXPECT_THAT(MultiscaleMetadata::FromJson(
                  GetMetadata("uint8", ScaleMetadata::Encoding::jpeg, 1, -5)),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*-5.*"));
  EXPECT_THAT(MultiscaleMetadata::FromJson(
                  GetMetadata("uint8", ScaleMetadata::Encoding::jpeg, 1, 101)),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*101.*"));

  // Test that jpeg_quality is valid for `jpeg` encoding.
  for (int quality : {0, 50, 100}) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto m, MultiscaleMetadata::FromJson(GetMetadata(
                    "uint8", ScaleMetadata::Encoding::jpeg, 1, quality)));
    ASSERT_EQ(1, m.scales.size());
    EXPECT_EQ(quality, m.scales[0].jpeg_quality);
  }

  // Test invalid number of channels for `jpeg` encoding.
  for (int num_channels : {2, 4, 5}) {
    EXPECT_THAT(MultiscaleMetadata::FromJson(GetMetadata(
                    "uint8", ScaleMetadata::Encoding::jpeg, num_channels)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test invalid data types for `jpeg` encoding.
  for (auto data_type_id :
       {DataTypeId::uint16_t, DataTypeId::uint32_t, DataTypeId::uint64_t,
        DataTypeId::int8_t, DataTypeId::int16_t, DataTypeId::int32_t,
        DataTypeId::int64_t, DataTypeId::float16_t, DataTypeId::float32_t,
        DataTypeId::float64_t, DataTypeId::complex64_t,
        DataTypeId::complex128_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    EXPECT_THAT(MultiscaleMetadata::FromJson(
                    GetMetadata(dtype.name(), ScaleMetadata::Encoding::jpeg)),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Test valid data types for `compressed_segmentation` encoding.
  for (auto data_type_id : {DataTypeId::uint32_t, DataTypeId::uint64_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto m,
        MultiscaleMetadata::FromJson(GetMetadata(
            dtype.name(), ScaleMetadata::Encoding::compressed_segmentation)));
    ASSERT_EQ(1, m.scales.size());
    EXPECT_EQ(dtype, m.dtype);
    EXPECT_EQ(ScaleMetadata::Encoding::compressed_segmentation,
              m.scales[0].encoding);
  }

  // Test invalid data types for `compressed_segmentation` encoding.
  for (auto data_type_id :
       {DataTypeId::uint8_t, DataTypeId::uint16_t, DataTypeId::int8_t,
        DataTypeId::int16_t, DataTypeId::int32_t, DataTypeId::int64_t,
        DataTypeId::float16_t, DataTypeId::float32_t, DataTypeId::float64_t,
        DataTypeId::complex64_t, DataTypeId::complex128_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    EXPECT_THAT(
        MultiscaleMetadata::FromJson(GetMetadata(
            dtype.name(), ScaleMetadata::Encoding::compressed_segmentation)),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

TEST(MetadataTest, ParseInvalid) {
  EXPECT_THAT(MultiscaleMetadata::FromJson(3),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  ::nlohmann::json metadata_json{{"@type", "neuroglancer_multiscale_volume"},
                                 {"num_channels", 1},
                                 {"scales",
                                  {{{"chunk_sizes", {{64, 65, 66}}},
                                    {"encoding", "raw"},
                                    {"key", "8_8_8"},
                                    {"resolution", {5, 6, 7}},
                                    {"size", {6446, 6643, 8090}},
                                    {"voxel_offset", {2, 4, 6}}}}},
                                 {"type", "segmentation"},
                                 {"data_type", "uint8"}};

  // Tests that setting any of the following members to null triggers an error.
  for (const char* k :
       {"@type", "num_channels", "type", "scales", "data_type"}) {
    auto invalid_json = metadata_json;
    invalid_json[k] = nullptr;
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")));
  }
  // Tests that missing any of the following members triggers an error.
  for (const char* k : {"num_channels", "type", "scales", "data_type"}) {
    auto invalid_json = metadata_json;
    invalid_json.erase(k);
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")));
  }
  // Tests that setting any of the following members to "invalid_string"
  // triggers an error.
  for (const char* k : {"@type", "num_channels", "scales", "data_type"}) {
    auto invalid_json = metadata_json;
    invalid_json[k] = "invalid_string";
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*invalid_string.*")));
  }
  // Tests that setting any of the following scale members to null triggers an
  // error.
  for (const char* k : {"chunk_sizes", "encoding", "key", "resolution", "size",
                        "voxel_offset"}) {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0][k] = nullptr;
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")))
        << k;
  }
  // Tests that setting any of the following to an array of length 2 triggers an
  // error.
  for (const char* k : {"resolution", "size", "voxel_offset"}) {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0][k] = {2, 3};
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")));
  }

  // Tests that a negative size trigger an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["size"] = {-1, 2, 7};
    EXPECT_THAT(
        MultiscaleMetadata::FromJson(invalid_json),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"size\".*"));
  }

  // Tests that invalid bounds trigger an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["scales"][0]["voxel_offset"] =  //
        {tensorstore::kMaxFiniteIndex, 2, 7};
    EXPECT_THAT(MultiscaleMetadata::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"voxel_offset\".*"));
  }
}

TEST(MultiscaleMetadataConstraintsTest, ParseEmptyObject) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m,
      MultiscaleMetadataConstraints::FromJson(::nlohmann::json::object_t{}));
  EXPECT_FALSE(m.type);
  EXPECT_FALSE(m.dtype.valid());
  EXPECT_FALSE(m.num_channels);
}

TEST(MultiscaleMetadataConstraintsTest, ParseValid) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m,
      MultiscaleMetadataConstraints::FromJson(
          {{"data_type", "uint8"}, {"num_channels", 3}, {"type", "image"}}));
  EXPECT_EQ("image", m.type.value());
  EXPECT_EQ(dtype_v<std::uint8_t>, m.dtype);
  EXPECT_EQ(3, m.num_channels.value());
}

TEST(MultiscaleMetadataConstraintsTest, ParseInvalid) {
  for (const char* k : {"data_type", "num_channels", "type"}) {
    ::nlohmann::json j{
        {"data_type", "uint8"}, {"num_channels", 3}, {"type", "image"}};
    j[k] = nullptr;
    EXPECT_THAT(MultiscaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")));
  }
  EXPECT_THAT(
      MultiscaleMetadataConstraints::FromJson(
          {{"extra", "member"}, {"data_type", "uint8"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"extra\".*"));
}

TEST(ScaleMetadataConstraintsTest, ParseEmptyObject) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, ScaleMetadataConstraints::FromJson(::nlohmann::json::object_t{}));
  EXPECT_FALSE(m.key);
  EXPECT_FALSE(m.box);
  EXPECT_FALSE(m.chunk_size);
  EXPECT_FALSE(m.resolution);
  EXPECT_FALSE(m.encoding);
  EXPECT_FALSE(m.jpeg_quality);
  EXPECT_FALSE(m.compressed_segmentation_block_size);
  EXPECT_FALSE(m.sharding);
}

TEST(ScaleMetadataConstraintsTest, ParseValid) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, ScaleMetadataConstraints::FromJson(
                  {{"key", "k"},
                   {"size", {1, 2, 3}},
                   {"voxel_offset", {4, 5, 6}},
                   {"resolution", {5, 6, 7}},
                   {"chunk_size", {2, 3, 4}},
                   {"encoding", "compressed_segmentation"},
                   {"compressed_segmentation_block_size", {4, 5, 6}},
                   {"sharding",
                    {{"@type", "neuroglancer_uint64_sharded_v1"},
                     {"preshift_bits", 1},
                     {"minishard_bits", 2},
                     {"shard_bits", 3},
                     {"hash", "identity"}}}}));
  EXPECT_THAT(m.key, ::testing::Optional(::testing::Eq("k")));
  EXPECT_THAT(m.box, ::testing::Optional(Box({4, 5, 6}, {1, 2, 3})));
  EXPECT_THAT(m.resolution, ::testing::Optional(ElementsAre(5, 6, 7)));
  EXPECT_THAT(m.chunk_size, ::testing::Optional(ElementsAre(2, 3, 4)));
  EXPECT_THAT(
      m.encoding,
      ::testing::Optional(ScaleMetadata::Encoding::compressed_segmentation));
  EXPECT_THAT(m.compressed_segmentation_block_size,
              ::testing::Optional(ElementsAre(4, 5, 6)));
  EXPECT_THAT(m.sharding,
              ::testing::Optional(ShardingSpec{
                  /*.hash_function=*/ShardingSpec::HashFunction::identity,
                  /*.preshift_bits=*/1,
                  /*.minishard_bits=*/2,
                  /*.shard_bits=*/3,
                  /*data_encoding=*/ShardingSpec::DataEncoding::raw,
                  /*minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
              }));
}

TEST(ScaleMetadataConstraintsTest, ParseValidNullSharding) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, ScaleMetadataConstraints::FromJson(
                  {{"key", "k"},
                   {"size", {1, 2, 3}},
                   {"voxel_offset", {4, 5, 6}},
                   {"resolution", {5, 6, 7}},
                   {"chunk_size", {2, 3, 4}},
                   {"encoding", "compressed_segmentation"},
                   {"compressed_segmentation_block_size", {4, 5, 6}},
                   {"sharding", nullptr}}));
  ASSERT_TRUE(m.sharding);
  EXPECT_TRUE(std::holds_alternative<NoShardingSpec>(*m.sharding));
}

TEST(ScaleMetadataConstraintsTest, ParseInvalid) {
  ::nlohmann::json metadata_json_jpeg{
      {"key", "k"},
      {"size", {1, 2, 3}},
      {"voxel_offset", {4, 5, 6}},
      {"resolution", {5, 6, 7}},
      {"chunk_size", {2, 3, 4}},
      {"encoding", "jpeg"},
      {"sharding",
       {{"@type", "neuroglancer_uint64_sharded_v1"},
        {"preshift_bits", 1},
        {"minishard_bits", 2},
        {"shard_bits", 3},
        {"hash", "identity"}}}};

  auto metadata_json_cseg = metadata_json_jpeg;
  metadata_json_cseg["encoding"] = "compressed_segmentation";
  metadata_json_cseg["compressed_segmentation_block_size"] = {4, 5, 6};

  // Control cases
  {
    auto with_quality = metadata_json_jpeg;
    with_quality["jpeg_quality"] = 70;
    TENSORSTORE_EXPECT_OK(ScaleMetadataConstraints::FromJson(with_quality));
  }

  TENSORSTORE_EXPECT_OK(ScaleMetadataConstraints::FromJson(metadata_json_jpeg));
  TENSORSTORE_EXPECT_OK(ScaleMetadataConstraints::FromJson(metadata_json_cseg));

  // Tests that an incompatible encoding triggers an error.
  EXPECT_THAT(OpenConstraints::FromJson(
                  {{"scale_metadata", metadata_json_cseg},
                   {"multiscale_metadata", {{"data_type", "uint8"}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*uint8.*"));

  // Tests that an incompatible number of channels triggers an error.
  {
    EXPECT_THAT(OpenConstraints::FromJson(
                    {{"scale_metadata", metadata_json_jpeg},
                     {"multiscale_metadata", {{"num_channels", 12345}}}}),
                MatchesStatus(absl::StatusCode::kInvalidArgument, ".*12345.*"));
  }

  // Tests that `compressed_segmentation_block_size` must not be specified with
  // an encoding of `raw` or with an unspecified encoding.
  {
    auto j = metadata_json_cseg;
    j["encoding"] = "raw";
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"compressed_segmentation_block_size\".*"));
    j.erase("encoding");
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"compressed_segmentation_block_size\".*"));
  }

  // Tests that `jpeg_quality` must not be specified with an encoding of `raw`
  // or with an unspecified encoding.
  {
    auto j = metadata_json_jpeg;
    j["encoding"] = "raw";
    j["jpeg_quality"] = 70;
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"jpeg_quality\".*"));
    j.erase("encoding");
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"jpeg_quality\".*"));
  }

  // Tests that setting any of the following members to null triggers an error.
  for (const char* k :
       {"key", "size", "voxel_offset", "resolution", "chunk_size", "encoding",
        "compressed_segmentation_block_size"}) {
    auto j = metadata_json_cseg;
    j[k] = nullptr;
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              StrCat(".*\"", k, "\".*")));
  }
  // Tests that an extra member triggers an error.
  EXPECT_THAT(
      ScaleMetadataConstraints::FromJson({{"extra", "member"}, {"key", "k"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"extra\".*"));

  // Tests that `voxel_offset` must not be specified without `size`.
  {
    auto j = metadata_json_jpeg;
    j.erase("size");
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"voxel_offset\".*"));
  }

  // Tests that invalid bounds trigger an error.
  {
    auto j = metadata_json_jpeg;
    j["voxel_offset"] = {2, tensorstore::kMaxFiniteIndex, 3};
    EXPECT_THAT(ScaleMetadataConstraints::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"voxel_offset\".*"));
  }

  // Tests that an invalid sharding spec triggers an error.
  {
    auto j = metadata_json_jpeg;
    j["sharding"] = "invalid";
    EXPECT_THAT(
        ScaleMetadataConstraints::FromJson(j),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"sharding\".*"));
  }

  // Tests that specifying `"sharding"` with a chunk size such that the
  // resultant compressed z index keys would exceed 64 bits results in an error.
  {
    auto j = metadata_json_jpeg;
    j["size"] = {0xffffffff, 0xffffffff, 0xffffffff};
    j["chunk_size"] = {1, 1, 1};
    EXPECT_THAT(
        ScaleMetadataConstraints::FromJson(j),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            "\"size\" of .* with \"chunk_size\" of .* is not compatible with "
            "sharded format because the chunk keys would exceed 64 bits"));

    // Verify that error does not occur when `"sharding"` is not specified.
    j["sharding"] = nullptr;
    TENSORSTORE_EXPECT_OK(ScaleMetadataConstraints::FromJson(j));
  }
}

TEST(OpenConstraintsTest, ParseEmptyObject) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, OpenConstraints::FromJson(::nlohmann::json::object_t{}));
  EXPECT_FALSE(m.scale_index);
}

TEST(OpenConstraintsTest, ParseEmptyObjectDataTypeConstraint) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, OpenConstraints::FromJson(::nlohmann::json::object_t{},
                                        dtype_v<std::uint8_t>));
  EXPECT_FALSE(m.scale_index);
  EXPECT_EQ(dtype_v<std::uint8_t>, m.multiscale.dtype);
}

TEST(OpenConstraintsTest, ParseEmptyObjectInvalidDataTypeConstraint) {
  EXPECT_THAT(
      OpenConstraints::FromJson(::nlohmann::json::object_t{}, dtype_v<bool>),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "bool data type is not one of the supported data types: .*"));
}

TEST(OpenConstraintsTest, ParseValid) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, OpenConstraints::FromJson(
                  {{"multiscale_metadata", {{"data_type", "uint8"}}},
                   {"scale_metadata", {{"encoding", "jpeg"}}},
                   {"scale_index", 2}}));
  EXPECT_THAT(m.scale_index, ::testing::Optional(2));
  EXPECT_EQ(dtype_v<std::uint8_t>, m.multiscale.dtype);
  EXPECT_EQ(ScaleMetadata::Encoding::jpeg, m.scale.encoding);
}

TEST(OpenConstraintsTest, ParseDataTypeConstraint) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto m, OpenConstraints::FromJson(
                  {{"multiscale_metadata", {{"data_type", "uint8"}}}},
                  dtype_v<std::uint8_t>));
  EXPECT_EQ(dtype_v<std::uint8_t>, m.multiscale.dtype);
}

TEST(OpenConstraintsTest, ParseDataTypeConstraintMismatch) {
  EXPECT_THAT(
      OpenConstraints::FromJson(
          {{"multiscale_metadata", {{"data_type", "uint8"}}}},
          dtype_v<std::uint16_t>),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: Expected data type of uint16 but received: uint8"));
}

TEST(OpenConstraintsTest, ParseInvalid) {
  ::nlohmann::json metadata_json{
      {"multiscale_metadata", {{"data_type", "uint8"}}},
      {"scale_metadata", {{"encoding", "jpeg"}}},
      {"scale_index", 2}};

  // Tests that an invalid `scale_index` results in an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["scale_index"] = -1;
    EXPECT_THAT(OpenConstraints::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"scale_index\".*"));
  }

  // Tests that an invalid `scale_metadata` results in an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["scale_metadata"] = 3;
    EXPECT_THAT(OpenConstraints::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"scale_metadata\".*"));
  }

  // Tests that an invalid `multiscale_metadata` results in an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["multiscale_metadata"] = 3;
    EXPECT_THAT(OpenConstraints::FromJson(invalid_json),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*\"multiscale_metadata\".*"));
  }

  // Tests that a `scale_metadata` incompatible with the `multiscale_metadata`
  // results in an error.
  {
    auto invalid_json = metadata_json;
    invalid_json["multiscale_metadata"]["num_channels"] = 2;
    EXPECT_THAT(
        OpenConstraints::FromJson(invalid_json),
        MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"jpeg\".*"));
  }
}

TEST(ValidateMetadataCompatibilityTest, Basic) {
  ::nlohmann::json metadata_json{
      {"data_type", "uint64"},
      {"num_channels", 1},
      {"scales",
       {
           {
               {"chunk_sizes", {{64, 65, 66}}},
               {"encoding", "compressed_segmentation"},
               {"compressed_segmentation_block_size", {8, 9, 10}},
               {"key", "8_8_8"},
               {"resolution", {5, 6, 7}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
               {"sharding",
                {{"@type", "neuroglancer_uint64_sharded_v1"},
                 {"data_encoding", "gzip"},
                 {"hash", "identity"},
                 {"minishard_bits", 6},
                 {"minishard_index_encoding", "gzip"},
                 {"preshift_bits", 9},
                 {"shard_bits", 11}}},
           },
           {
               {"chunk_sizes", {{8, 9, 10}, {11, 12, 13}}},
               {"encoding", "raw"},
               {"key", "16_16_16"},
               {"resolution", {10, 11, 12}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
           },
       }},
      {"type", "segmentation"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto a,
                                   MultiscaleMetadata::FromJson(metadata_json));

  const auto Validate = [](const MultiscaleMetadata& a,
                           const MultiscaleMetadata& b, std::size_t scale_index,
                           std::array<Index, 3> chunk_size) -> absl::Status {
    SCOPED_TRACE(StrCat("a=", ::nlohmann::json(a).dump()));
    SCOPED_TRACE(StrCat("b=", ::nlohmann::json(b).dump()));
    SCOPED_TRACE(StrCat("scale_index=", scale_index));
    SCOPED_TRACE(StrCat("chunk_size=", ::nlohmann::json(chunk_size).dump()));
    auto status = ValidateMetadataCompatibility(a, b, scale_index, chunk_size);
    auto key_a = GetMetadataCompatibilityKey(a, scale_index, chunk_size);
    auto key_b = GetMetadataCompatibilityKey(b, scale_index, chunk_size);
    if (status.ok()) {
      EXPECT_EQ(key_a, key_b);
    } else {
      EXPECT_NE(key_a, key_b);
    }
    return status;
  };

  // Identical metadata is always compatible.
  TENSORSTORE_EXPECT_OK(Validate(a, a, 0, {{64, 65, 66}}));
  TENSORSTORE_EXPECT_OK(Validate(a, a, 1, {{8, 9, 10}}));
  TENSORSTORE_EXPECT_OK(Validate(a, a, 1, {{11, 12, 13}}));

  {
    SCOPED_TRACE("Other scales can be removed");
    auto b = a;
    b.scales.resize(1);
    TENSORSTORE_EXPECT_OK(Validate(a, b, 0, {{64, 65, 66}}));
  }

  {
    SCOPED_TRACE("Other scales can be added");
    auto b = a;
    b.scales.push_back(b.scales[1]);
    TENSORSTORE_EXPECT_OK(Validate(a, b, 0, {{64, 65, 66}}));
  }

  {
    SCOPED_TRACE("Other scales can change");
    auto b = a;
    b.scales[1].key = "new_key";
    TENSORSTORE_EXPECT_OK(Validate(a, b, 0, {{64, 65, 66}}));
  }

  {
    SCOPED_TRACE("`data_type` cannot change");
    auto b = a;
    b.dtype = dtype_v<std::int16_t>;
    EXPECT_THAT(Validate(a, b, 0, {{64, 65, 66}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"data_type\".*"));
  }

  {
    SCOPED_TRACE("`num_channels` cannot change");
    auto b = a;
    b.num_channels = 3;
    EXPECT_THAT(Validate(a, b, 0, {{64, 65, 66}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"num_channels\".*"));
  }

  {
    SCOPED_TRACE("`type` can change");
    auto b = a;
    b.type = "image";
    TENSORSTORE_EXPECT_OK(Validate(a, b, 0, {{64, 65, 66}}));
  }

  {
    SCOPED_TRACE("`resolution` can change");
    auto b = a;
    b.scales[0].resolution[0] = 42;
    TENSORSTORE_EXPECT_OK(Validate(a, b, 0, {{64, 65, 66}}));
  }

  {
    SCOPED_TRACE("`size` cannot change");
    auto b = a;
    b.scales[0].box.shape()[0] = 42;
    EXPECT_THAT(
        Validate(a, b, 0, {{64, 65, 66}}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"size\".*"));
  }

  {
    SCOPED_TRACE("`voxel_offset` cannot change");
    auto b = a;
    b.scales[0].box.origin()[0] = 42;
    EXPECT_THAT(Validate(a, b, 0, {{64, 65, 66}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"voxel_offset\".*"));
  }

  {
    SCOPED_TRACE("`encoding` cannot change");
    auto b = a;
    b.scales[1].encoding = ScaleMetadata::Encoding::jpeg;
    EXPECT_THAT(Validate(a, b, 1, {{8, 9, 10}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"encoding\".*"));
  }

  {
    SCOPED_TRACE(
        "The new `chunk_sizes` must include the specified `chunk_size`");
    auto b = a;
    b.scales[1].chunk_sizes = {{{6, 7, 8}}, {{8, 9, 10}}};
    // Do not test `GetMetadataCompatibilityKey` in this case.
    EXPECT_THAT(ValidateMetadataCompatibility(a, b, 1, {{11, 12, 13}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\\[11,12,13\\].*"));
    TENSORSTORE_EXPECT_OK(Validate(a, b, 1, {{8, 9, 10}}));
  }

  {
    SCOPED_TRACE("`key` must not change");
    auto b = a;
    b.scales[0].key = "new_key";
    EXPECT_THAT(
        Validate(a, b, 0, {{64, 65, 66}}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"key\".*"));
  }

  {
    SCOPED_TRACE("scale must be present");
    auto b = a;
    b.scales.resize(1);
    // Do not test `GetMetadataCompatibilityKey` in this case.
    EXPECT_THAT(ValidateMetadataCompatibility(a, b, 1, {{8, 9, 10}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".* missing scale 1"));
  }

  {
    SCOPED_TRACE("`sharding` must not change");
    auto b = a;
    b.scales[0].sharding = NoShardingSpec{};
    EXPECT_THAT(Validate(a, b, 0, {{64, 65, 66}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"sharding\".*"));
  }

  {
    SCOPED_TRACE("`compressed_segmentation_block_size` must not change");
    auto b = a;
    b.scales[0].compressed_segmentation_block_size[0] = 42;
    EXPECT_THAT(Validate(a, b, 0, {{64, 65, 66}}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              ".*\"compressed_segmentation_block_size\".*"));
  }
}

TEST(CreateScaleTest, NoExistingMetadata) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   OpenConstraints::FromJson({
                                       {"multiscale_metadata",
                                        {
                                            {"data_type", "uint8"},
                                            {"num_channels", 2},
                                            {"type", "image"},
                                        }},
                                       {"scale_metadata",
                                        {
                                            {"encoding", "raw"},
                                            {"key", "scale_key"},
                                            {"size", {10, 11, 12}},
                                            {"voxel_offset", {1, 2, 3}},
                                            {"resolution", {5, 6, 7}},
                                            {"chunk_size", {8, 9, 10}},
                                        }},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, CreateScale(/*existing_metadata=*/nullptr, constraints,
                               /*schema=*/{}));
  const auto& [metadata, scale_index] = result;
  ASSERT_TRUE(metadata);
  EXPECT_EQ(0, scale_index);
  const ::nlohmann::json scale_attributes{
      {"encoding", "raw"},           {"key", "scale_key"},
      {"size", {10, 11, 12}},        {"voxel_offset", {1, 2, 3}},
      {"chunk_sizes", {{8, 9, 10}}}, {"resolution", {5, 6, 7}}};
  EXPECT_EQ(::nlohmann::json::object_t(), metadata->extra_attributes);
  EXPECT_THAT(metadata->ToJson(),
              ::testing::Optional(
                  MatchesJson({{"@type", "neuroglancer_multiscale_volume"},
                               {"type", "image"},
                               {"data_type", "uint8"},
                               {"num_channels", 2},
                               {"scales", {scale_attributes}}})));
  EXPECT_EQ(dtype_v<std::uint8_t>, metadata->dtype);
  EXPECT_EQ(2, metadata->num_channels);
  EXPECT_EQ("image", metadata->type);
  ASSERT_EQ(1, metadata->scales.size());
  auto& s = metadata->scales[0];
  EXPECT_EQ("scale_key", s.key);
  EXPECT_EQ(s.box, Box({1, 2, 3}, {10, 11, 12}));
  EXPECT_THAT(s.chunk_sizes, ElementsAre(ElementsAre(8, 9, 10)));
  EXPECT_THAT(s.resolution, ElementsAre(5, 6, 7));
  EXPECT_EQ(ScaleMetadata::Encoding::raw, s.encoding);
  EXPECT_EQ(::nlohmann::json::object_t(), s.extra_attributes);
  EXPECT_THAT(s.ToJson(), ::testing::Optional(MatchesJson(scale_attributes)));
}

// Tests that the scale key is generated from the `resolution` if it is not
// specified.
TEST(CreateScaleTest, NoExistingMetadataGenerateKey) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   OpenConstraints::FromJson({
                                       {"multiscale_metadata",
                                        {
                                            {"data_type", "uint8"},
                                            {"num_channels", 2},
                                            {"type", "image"},
                                        }},
                                       {"scale_metadata",
                                        {
                                            {"encoding", "raw"},
                                            {"size", {10, 11, 12}},
                                            {"voxel_offset", {1, 2, 3}},
                                            {"resolution", {5, 6, 7}},
                                            {"chunk_size", {8, 9, 10}},
                                        }},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      CreateScale(/*existing_metadata=*/nullptr, constraints, /*schema=*/{}));
  const auto& [metadata, scale_index] = result;
  ASSERT_TRUE(metadata);
  EXPECT_EQ(0, scale_index);
  const ::nlohmann::json scale_attributes{
      {"encoding", "raw"},           {"key", "5_6_7"},
      {"size", {10, 11, 12}},        {"voxel_offset", {1, 2, 3}},
      {"chunk_sizes", {{8, 9, 10}}}, {"resolution", {5, 6, 7}}};
  EXPECT_THAT(metadata->ToJson(),
              ::testing::Optional(
                  MatchesJson({{"@type", "neuroglancer_multiscale_volume"},
                               {"type", "image"},
                               {"data_type", "uint8"},
                               {"num_channels", 2},
                               {"scales", {scale_attributes}}})));
}

TEST(CreateScaleTest, NoExistingMetadataCompressedSegmentation) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto constraints,
      OpenConstraints::FromJson(
          {{"multiscale_metadata",
            {
                {"data_type", "uint32"},
                {"num_channels", 2},
                {"type", "image"},
            }},
           {"scale_metadata",
            {
                {"encoding", "compressed_segmentation"},
                {"compressed_segmentation_block_size", {8, 9, 10}},
                {"key", "scale_key"},
                {"size", {10, 11, 12}},
                {"voxel_offset", {1, 2, 3}},
                {"resolution", {5, 6, 7}},
                {"chunk_size", {8, 9, 10}},
            }},
           {"scale_index", 0}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      CreateScale(/*existing_metadata=*/nullptr, constraints, /*schema=*/{}));
  const auto& [metadata, scale_index] = result;
  ASSERT_TRUE(metadata);
  EXPECT_EQ(0, scale_index);
  const ::nlohmann::json scale_attributes{
      {"encoding", "compressed_segmentation"},
      {"compressed_segmentation_block_size", {8, 9, 10}},
      {"key", "scale_key"},
      {"size", {10, 11, 12}},
      {"voxel_offset", {1, 2, 3}},
      {"chunk_sizes", {{8, 9, 10}}},
      {"resolution", {5, 6, 7}}};
  EXPECT_THAT(metadata->ToJson(),
              ::testing::Optional(
                  MatchesJson({{"@type", "neuroglancer_multiscale_volume"},
                               {"type", "image"},
                               {"data_type", "uint32"},
                               {"num_channels", 2},
                               {"scales", {scale_attributes}}})));
  EXPECT_EQ(dtype_v<std::uint32_t>, metadata->dtype);
  EXPECT_EQ(2, metadata->num_channels);
  EXPECT_EQ("image", metadata->type);
  ASSERT_EQ(1, metadata->scales.size());
  auto& s = metadata->scales[0];
  EXPECT_EQ("scale_key", s.key);
  EXPECT_EQ(s.box, Box({1, 2, 3}, {10, 11, 12}));
  EXPECT_THAT(s.chunk_sizes, ElementsAre(ElementsAre(8, 9, 10)));
  EXPECT_THAT(s.resolution, ElementsAre(5, 6, 7));
  EXPECT_EQ(ScaleMetadata::Encoding::compressed_segmentation, s.encoding);
  EXPECT_THAT(s.compressed_segmentation_block_size, ElementsAre(8, 9, 10));
  EXPECT_THAT(s.ToJson(), ::testing::Optional(MatchesJson(scale_attributes)));
}

TEST(CreateScaleTest, InvalidScaleConstraints) {
  ::nlohmann::json constraints_json{
      {"multiscale_metadata",
       {
           {"data_type", "uint32"},
           {"num_channels", 2},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"key", "scale_key"},
           {"size", {10, 11, 12}},
           {"voxel_offset", {1, 2, 3}},
           {"resolution", {5, 6, 7}},
           {"sharding", nullptr},
           {"chunk_size", {8, 9, 10}},
       }}};
  // Control case
  TENSORSTORE_EXPECT_OK(CreateScale(
      /*existing_metadata=*/nullptr,
      OpenConstraints::FromJson(constraints_json).value(),
      /*schema=*/{}));

  // Tests that create fails when a non-zero "scale_index" is specified.
  {
    auto j = constraints_json;
    j["scale_index"] = 1;
    EXPECT_THAT(
        CreateScale(/*existing_metadata=*/nullptr,
                    OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      ".*Cannot create scale 1 in new multiscale volume"));
  }
}

TEST(CreateScaleTest, InvalidMultiscaleConstraints) {
  ::nlohmann::json constraints_json{
      {"multiscale_metadata",
       {
           {"data_type", "uint32"},
           {"num_channels", 2},
           {"type", "image"},
       }},
      {"scale_metadata",
       {
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"key", "scale_key"},
           {"size", {10, 11, 12}},
           {"voxel_offset", {1, 2, 3}},
           {"resolution", {5, 6, 7}},
           {"sharding", nullptr},
           {"chunk_size", {8, 9, 10}},
       }}};
  // Control case
  TENSORSTORE_EXPECT_OK(CreateScale(
      /*existing_metadata=*/nullptr,
      OpenConstraints::FromJson(constraints_json).value(),
      /*schema=*/{}));

  // Tests that removing any of the following keys results in an error.
  for (const char* k : {"data_type", "num_channels"}) {
    auto j = constraints_json;
    j["multiscale_metadata"].erase(k);
    EXPECT_THAT(CreateScale(/*existing_metadata=*/nullptr,
                            OpenConstraints::FromJson(j).value(),
                            /*schema=*/{}),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }
}

TEST(CreateScaleTest, ExistingMetadata) {
  ::nlohmann::json constraints_json{
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 1},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"key", "scale_key"},
           {"size", {10, 11, 12}},
           {"voxel_offset", {1, 2, 3}},
           {"resolution", {5, 6, 7}},
           {"sharding", nullptr},
           {"chunk_size", {8, 9, 10}},
       }}};
  ::nlohmann::json metadata_json{
      {"data_type", "uint64"},
      {"num_channels", 1},
      {"scales",
       {
           {
               {"chunk_sizes", {{64, 65, 66}}},
               {"encoding", "compressed_segmentation"},
               {"compressed_segmentation_block_size", {8, 9, 10}},
               {"key", "8_8_8"},
               {"resolution", {5, 6, 7}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
               {"sharding",
                {{"@type", "neuroglancer_uint64_sharded_v1"},
                 {"data_encoding", "gzip"},
                 {"hash", "identity"},
                 {"minishard_bits", 6},
                 {"minishard_index_encoding", "gzip"},
                 {"preshift_bits", 9},
                 {"shard_bits", 11}}},
           },
           {
               {"chunk_sizes", {{8, 9, 10}, {11, 12, 13}}},
               {"encoding", "raw"},
               {"key", "16_16_16"},
               {"resolution", {10, 11, 12}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
           },
       }},
      {"type", "segmentation"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto existing_metadata,
                                   MultiscaleMetadata::FromJson(metadata_json));
  auto expected_metadata = metadata_json;
  const ::nlohmann::json scale_attributes{
      {"encoding", "compressed_segmentation"},
      {"compressed_segmentation_block_size", {8, 9, 10}},
      {"key", "scale_key"},
      {"size", {10, 11, 12}},
      {"voxel_offset", {1, 2, 3}},
      {"chunk_sizes", {{8, 9, 10}}},
      {"resolution", {5, 6, 7}}};
  expected_metadata["@type"] = "neuroglancer_multiscale_volume";
  expected_metadata["scales"].push_back(scale_attributes);

  // Test with full set of constraints.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto constraints, OpenConstraints::FromJson(constraints_json));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto result,
        CreateScale(&existing_metadata, constraints, /*schema=*/{}));
    const auto& [metadata, scale_index] = result;
    ASSERT_TRUE(metadata);
    EXPECT_EQ(2, scale_index);
    EXPECT_THAT(metadata->ToJson(),
                ::testing::Optional(MatchesJson(expected_metadata)));
    EXPECT_EQ(dtype_v<std::uint64_t>, metadata->dtype);
    EXPECT_EQ(1, metadata->num_channels);
    EXPECT_EQ("segmentation", metadata->type);
    ASSERT_EQ(3, metadata->scales.size());
    auto& s = metadata->scales[2];
    EXPECT_EQ("scale_key", s.key);
    EXPECT_EQ(s.box, Box({1, 2, 3}, {10, 11, 12}));
    EXPECT_THAT(s.chunk_sizes, ElementsAre(ElementsAre(8, 9, 10)));
    EXPECT_THAT(s.resolution, ElementsAre(5, 6, 7));
    EXPECT_EQ(ScaleMetadata::Encoding::compressed_segmentation, s.encoding);
    EXPECT_THAT(s.compressed_segmentation_block_size, ElementsAre(8, 9, 10));
    EXPECT_THAT(s.ToJson(), ::testing::Optional(MatchesJson(scale_attributes)));
  }

  // Test that `scale_index` may be specified.
  {
    auto j = constraints_json;
    j["scale_index"] = 2;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto result,
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}));
    const auto& [metadata, scale_index] = result;
    ASSERT_TRUE(metadata);
    EXPECT_EQ(2, scale_index);
    EXPECT_THAT(metadata->ToJson(),
                ::testing::Optional(MatchesJson(expected_metadata)));
  }

  // Test that `key` may be unspecified.
  {
    auto j = constraints_json;
    j["scale_metadata"].erase("key");
    j["scale_metadata"]["resolution"] = {41, 42, 43};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto result,
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}));
    const auto& [metadata, scale_index] = result;
    ASSERT_TRUE(metadata);
    EXPECT_EQ(2, scale_index);
    auto expected = expected_metadata;
    expected["scales"][2]["key"] = "41_42_43";
    expected["scales"][2]["resolution"] = {41, 42, 43};
    EXPECT_THAT(metadata->ToJson(), ::testing::Optional(MatchesJson(expected)));
  }

  // Test that any of the following `multiscale_metadata` keys may be omitted
  // without affecting the result.
  for (const char* k : {"data_type", "num_channels", "type"}) {
    auto j = constraints_json;
    j["multiscale_metadata"].erase(k);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto result,
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}));
    const auto& [metadata, scale_index] = result;
    ASSERT_TRUE(metadata);
    EXPECT_EQ(2, scale_index);
    EXPECT_THAT(metadata->ToJson(),
                ::testing::Optional(MatchesJson(expected_metadata)));
  }

  // Tests that any of the following changes in the `multiscale_metadata` leads
  // to an error.
  for (const auto& [k, new_value] :
       std::vector<std::pair<std::string, ::nlohmann::json>>{
           {"data_type", "uint32"}, {"num_channels", 3}, {"type", "image"}}) {
    auto j = constraints_json;
    j["multiscale_metadata"][k] = new_value;
    EXPECT_THAT(
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      StrCat(".*\"", k, "\".*")));
  }

  // Tests that a mismatch between the `encoding` specified in `constraints` and
  // the existing `dtype` leads to an error.
  {
    auto j = constraints_json;
    j["multiscale_metadata"].erase("data_type");
    j["scale_metadata"].erase("compressed_segmentation_block_size");
    j["scale_metadata"]["encoding"] = "jpeg";
    EXPECT_THAT(
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"jpeg\".*"));
  }
}

TEST(CreateScaleTest, ExistingScale) {
  ::nlohmann::json constraints_json{
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 1},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"key", "8_8_8"},
           {"resolution", {5, 6, 7}},
           {"size", {6446, 6643, 8090}},
           {"voxel_offset", {2, 4, 6}},
           {"chunk_size", {64, 65, 66}},
       }}};
  ::nlohmann::json metadata_json{
      {"data_type", "uint64"},
      {"num_channels", 1},
      {"scales",
       {
           {
               {"chunk_sizes", {{64, 65, 66}}},
               {"encoding", "compressed_segmentation"},
               {"compressed_segmentation_block_size", {8, 9, 10}},
               {"key", "8_8_8"},
               {"resolution", {5, 6, 7}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
           },
           {
               {"chunk_sizes", {{8, 9, 10}, {11, 12, 13}}},
               {"encoding", "raw"},
               {"key", "16_16_16"},
               {"resolution", {10, 11, 12}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
           },
       }},
      {"type", "segmentation"},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto existing_metadata,
                                   MultiscaleMetadata::FromJson(metadata_json));

  EXPECT_THAT(CreateScale(&existing_metadata,
                          OpenConstraints::FromJson(constraints_json).value(),
                          /*schema=*/{}),
              MatchesStatus(absl::StatusCode::kAlreadyExists,
                            "Scale with key \"8_8_8\" already exists"));

  {
    auto j = constraints_json;
    j["scale_index"] = 3;
    EXPECT_THAT(
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(absl::StatusCode::kFailedPrecondition,
                      "Scale index to create \\(3\\) must equal the "
                      "existing number of scales \\(2\\)"));
  }

  for (int scale_index = 0; scale_index < 2; ++scale_index) {
    auto j = constraints_json;
    j["scale_index"] = scale_index;
    EXPECT_THAT(
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(absl::StatusCode::kAlreadyExists,
                      StrCat("Scale index ", scale_index, " already exists")));
  }

  {
    auto j = constraints_json;
    j["scale_metadata"].erase("key");
    EXPECT_THAT(
        CreateScale(&existing_metadata, OpenConstraints::FromJson(j).value(),
                    /*schema=*/{}),
        MatchesStatus(
            absl::StatusCode::kAlreadyExists,
            "Scale with resolution \\[5\\.0,6\\.0,7\\.0\\] already exists"));
  }
}

class OpenScaleTest : public ::testing::Test {
 protected:
  ::nlohmann::json metadata_json{
      {"data_type", "uint64"},
      {"num_channels", 1},
      {"scales",
       {
           {{"chunk_sizes", {{64, 65, 66}}},
            {"encoding", "compressed_segmentation"},
            {"compressed_segmentation_block_size", {8, 9, 10}},
            {"key", "8_8_8"},
            {"resolution", {5, 6, 7}},
            {"size", {6446, 6643, 8090}},
            {"voxel_offset", {2, 4, 6}},
            {"sharding",
             {{"@type", "neuroglancer_uint64_sharded_v1"},
              {"data_encoding", "gzip"},
              {"hash", "identity"},
              {"minishard_bits", 6},
              {"minishard_index_encoding", "gzip"},
              {"preshift_bits", 9},
              {"shard_bits", 11}}}},
           {
               {"chunk_sizes", {{8, 9, 10}, {11, 12, 13}}},
               {"encoding", "raw"},
               {"key", "16_16_16"},
               {"resolution", {10, 11, 12}},
               {"size", {6446, 6643, 8090}},
               {"voxel_offset", {2, 4, 6}},
           },
       }},
      {"type", "segmentation"},
  };
  MultiscaleMetadata metadata =
      MultiscaleMetadata::FromJson(metadata_json).value();
};

TEST_F(OpenScaleTest, Success) {
  // Open with no constraints
  EXPECT_EQ(
      0u,
      OpenScale(metadata,
                OpenConstraints::FromJson(::nlohmann::json::object_t{}).value(),
                /*schema=*/{}));

  // Open by `scale_index` only
  EXPECT_EQ(0u,
            OpenScale(metadata,
                      OpenConstraints::FromJson({{"scale_index", 0}}).value(),
                      /*schema=*/{}));

  // Open with invalid `scale_index`
  EXPECT_THAT(OpenScale(metadata,
                        OpenConstraints::FromJson({{"scale_index", 2}}).value(),
                        /*schema=*/{}),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "Scale 2 does not exist, number of scales is 2"));

  // Open by `key` only
  EXPECT_EQ(0u, OpenScale(metadata,
                          OpenConstraints::FromJson(
                              {{"scale_metadata", {{"key", "8_8_8"}}}})
                              .value(),
                          /*schema=*/{}));
  EXPECT_EQ(1u, OpenScale(metadata,
                          OpenConstraints::FromJson(
                              {{"scale_metadata", {{"key", "16_16_16"}}}})
                              .value(),
                          /*schema=*/{}));

  // Open by `resolution` only
  EXPECT_EQ(0u, OpenScale(metadata,
                          OpenConstraints::FromJson(
                              {{"scale_metadata", {{"resolution", {5, 6, 7}}}}})
                              .value(),
                          /*schema=*/{}));
  EXPECT_EQ(1u,
            OpenScale(metadata,
                      OpenConstraints::FromJson(
                          {{"scale_metadata", {{"resolution", {10, 11, 12}}}}})
                          .value(),
                      /*schema=*/{}));

  // Open by `key` and `resolution`
  EXPECT_EQ(0u, OpenScale(metadata,
                          OpenConstraints::FromJson(
                              {{"scale_metadata",
                                {{"key", "8_8_8"}, {"resolution", {5, 6, 7}}}}})
                              .value(),
                          /*schema=*/{}));
}

TEST_F(OpenScaleTest, InvalidKey) {
  EXPECT_THAT(OpenScale(metadata,
                        OpenConstraints::FromJson(
                            {{"scale_metadata", {{"key", "invalidkey"}}}})
                            .value(),
                        /*schema=*/{}),
              MatchesStatus(absl::StatusCode::kNotFound,
                            "No scale found matching key=\"invalidkey\""));
}

TEST_F(OpenScaleTest, InvalidResolution) {
  EXPECT_THAT(
      OpenScale(metadata,
                OpenConstraints::FromJson(
                    {{"scale_metadata", {{"resolution", {41, 42, 43}}}}})
                    .value(),
                /*schema=*/{}),
      MatchesStatus(
          absl::StatusCode::kNotFound,
          "No scale found matching "
          "dimension_units=\\[\"41 nm\", \"42 nm\", \"43 nm\", null\\]"));
}

TEST_F(OpenScaleTest, InvalidKeyAndResolutionCombination) {
  EXPECT_THAT(
      OpenScale(metadata,
                OpenConstraints::FromJson(
                    {{"scale_metadata",
                      {{"key", "16_16_16"}, {"resolution", {5, 6, 7}}}}})
                    .value(),
                /*schema=*/{}),
      MatchesStatus(absl::StatusCode::kNotFound,
                    "No scale found matching "
                    "dimension_units=\\[\"5 nm\", \"6 nm\", \"7 nm\", null\\], "
                    "key=\"16_16_16\""));
}

TEST_F(OpenScaleTest, InvalidNumChannels) {
  EXPECT_THAT(OpenScale(metadata,
                        OpenConstraints::FromJson(
                            {{"scale_index", 0},
                             {"multiscale_metadata", {{"num_channels", 7}}}})
                            .value(),
                        /*schema=*/{}),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*\"num_channels\".*"));
}

TEST_F(OpenScaleTest, InvalidScaleSize) {
  EXPECT_THAT(
      OpenScale(
          metadata,
          OpenConstraints::FromJson(
              {{"scale_index", 0}, {"scale_metadata", {{"size", {1, 2, 3}}}}})
              .value(),
          /*schema=*/{}),
      MatchesStatus(absl::StatusCode::kFailedPrecondition, ".*\"size\".*"));
}

TEST_F(OpenScaleTest, MetadataMismatch) {
  ::nlohmann::json constraints_json{
      {"multiscale_metadata",
       {
           {"data_type", "uint64"},
           {"num_channels", 1},
           {"type", "segmentation"},
       }},
      {"scale_metadata",
       {
           {"encoding", "compressed_segmentation"},
           {"compressed_segmentation_block_size", {8, 9, 10}},
           {"key", "8_8_8"},
           {"resolution", {5, 6, 7}},
           {"size", {6446, 6643, 8090}},
           {"voxel_offset", {2, 4, 6}},
           {"chunk_size", {64, 65, 66}},
           {"sharding",
            {{"@type", "neuroglancer_uint64_sharded_v1"},
             {"data_encoding", "gzip"},
             {"hash", "identity"},
             {"minishard_bits", 6},
             {"minishard_index_encoding", "gzip"},
             {"preshift_bits", 9},
             {"shard_bits", 11}}},
       }},
      {"scale_index", 0},
  };
  // Control case.
  EXPECT_EQ(0u, OpenScale(metadata,
                          OpenConstraints::FromJson(constraints_json).value(),
                          /*schema=*/{}));

  // Tests that any of the following changes in the `scale_metadata` leads
  // to an error.
  for (const auto& [k, new_value] :
       std::vector<std::pair<std::string, ::nlohmann::json>>{
           {"encoding", "raw"},
           {"compressed_segmentation_block_size", {7, 8, 9}},
           {"key", "invalidkey"},
           {"resolution", {1, 2, 3}},
           {"size", {1, 2, 3}},
           {"voxel_offset", {0, 0, 0}},
           {"chunk_size", {4, 4, 4}},
           {"sharding", nullptr}}) {
    auto j = constraints_json;
    j["scale_metadata"][k] = new_value;
    if (k == "encoding") {
      j["scale_metadata"].erase("compressed_segmentation_block_size");
    }
    EXPECT_THAT(OpenScale(metadata, OpenConstraints::FromJson(j).value(),
                          /*schema=*/{}),
                MatchesStatus(absl::StatusCode::kFailedPrecondition,
                              StrCat(".*\"", k, "\".*")));
  }
}

TEST(ResolveScaleKeyTest, Basic) {
  EXPECT_EQ("a/b/c/d", ResolveScaleKey("a/b/c", "d"));
  EXPECT_EQ("a/b/d", ResolveScaleKey("a/b/c", "../d"));
  EXPECT_EQ("a/d", ResolveScaleKey("a/b/c", "../../d"));
  EXPECT_EQ("d", ResolveScaleKey("a/b/c", "../../../d"));
  EXPECT_EQ("../d", ResolveScaleKey("a/b/c", "../../../../d"));
}

TEST(ValidateDataTypeTest, Basic) {
  for (auto data_type_id :
       {DataTypeId::uint8_t, DataTypeId::uint16_t, DataTypeId::uint32_t,
        DataTypeId::uint64_t, DataTypeId::float32_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    TENSORSTORE_EXPECT_OK(ValidateDataType(dtype));
  }
  for (auto data_type_id :
       {DataTypeId::string_t, DataTypeId::json_t, DataTypeId::ustring_t,
        DataTypeId::bool_t, DataTypeId::float64_t, DataTypeId::complex64_t}) {
    const auto dtype = kDataTypes[static_cast<int>(data_type_id)];
    EXPECT_THAT(
        ValidateDataType(dtype),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            StrCat(dtype.name(),
                   " data type is not one of the supported data types: .*")));
  }
}

TEST(GetCompressedZIndexBitsTest, Basic) {
  EXPECT_THAT(
      GetCompressedZIndexBits(
          span<const Index, 3>({0, 0xffffffff, tensorstore::kMaxFiniteIndex}),
          span<const Index, 3>({20, 1, 1})),
      ::testing::ElementsAre(0, 32, 62));
  EXPECT_THAT(GetCompressedZIndexBits(span<const Index, 3>({79, 80, 144}),
                                      span<const Index, 3>({20, 20, 12})),
              ::testing::ElementsAre(2, 2, 4));
}

TEST(EncodeCompressedZIndexTest, Basic) {
  const std::array<int, 3> bits{{4, 2, 1}};
  EXPECT_EQ(0, EncodeCompressedZIndex(span<const Index, 3>({0, 0, 0}), bits));
  EXPECT_EQ(1, EncodeCompressedZIndex(span<const Index, 3>({1, 0, 0}), bits));
  EXPECT_EQ(2, EncodeCompressedZIndex(span<const Index, 3>({0, 1, 0}), bits));
  EXPECT_EQ(0b1100, EncodeCompressedZIndex(
                        span<const Index, 3>({0b10, 0b0, 0b1}), bits));
  EXPECT_EQ(0b11010, EncodeCompressedZIndex(
                         span<const Index, 3>({0b10, 0b11, 0b0}), bits));
  EXPECT_EQ(0b1010101, EncodeCompressedZIndex(
                           span<const Index, 3>({0b1001, 0b10, 0b1}), bits));
}

// Tests the simple case where all shards have the full number of chunks.
TEST(GetChunksPerVolumeShardFunctionTest, AllShardsFull) {
  ShardingSpec sharding_spec{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  const Index volume_shape[3] = {99, 98, 97};
  const Index chunk_shape[3] = {50, 25, 13};
  // Grid shape: {2, 4, 8}

  ShardChunkHierarchy hierarchy;
  ASSERT_TRUE(GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                                     hierarchy));
  EXPECT_THAT(hierarchy.z_index_bits, ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(hierarchy.grid_shape_in_chunks, ::testing::ElementsAre(2, 4, 8));
  EXPECT_THAT(hierarchy.minishard_shape_in_chunks,
              ::testing::ElementsAre(2, 1, 1));
  EXPECT_THAT(hierarchy.shard_shape_in_chunks, ::testing::ElementsAre(2, 2, 2));
  EXPECT_EQ(3, hierarchy.non_shard_bits);
  EXPECT_EQ(3, hierarchy.shard_bits);

  auto f =
      GetChunksPerVolumeShardFunction(sharding_spec, volume_shape, chunk_shape);

  // Shard shape in chunks: {2, 2, 2}
  for (std::uint64_t shard = 0; shard < 8; ++shard) {
    EXPECT_EQ(8, f(shard)) << "shard=" << shard;
  }

  // Invalid shards have 0 chunks.
  EXPECT_EQ(0, f(8));
}

// Tests the case where shards have varying number of chunks in 1 dimension.
TEST(GetChunksPerVolumeShardFunctionTest, PartialShards1Dim) {
  ShardingSpec sharding_spec{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  const Index volume_shape[3] = {99, 98, 90};
  const Index chunk_shape[3] = {50, 25, 13};
  // Grid shape: {2, 4, 7}
  // Full shard shape is {2, 2, 2}.

  ShardChunkHierarchy hierarchy;
  ASSERT_TRUE(GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                                     hierarchy));
  EXPECT_THAT(hierarchy.z_index_bits, ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(hierarchy.grid_shape_in_chunks, ::testing::ElementsAre(2, 4, 7));
  EXPECT_THAT(hierarchy.minishard_shape_in_chunks,
              ::testing::ElementsAre(2, 1, 1));
  EXPECT_THAT(hierarchy.shard_shape_in_chunks, ::testing::ElementsAre(2, 2, 2));
  EXPECT_EQ(3, hierarchy.non_shard_bits);
  EXPECT_EQ(3, hierarchy.shard_bits);

  auto f =
      GetChunksPerVolumeShardFunction(sharding_spec, volume_shape, chunk_shape);

  // Shard 0 has origin {0, 0, 0}
  EXPECT_EQ(8, f(0));

  // Shard 1 has origin {0, 2, 0}
  EXPECT_EQ(8, f(1));

  // Shard 2 has origin {0, 0, 2}
  EXPECT_EQ(8, f(2));

  // Shard 3 has origin {0, 2, 2}
  EXPECT_EQ(8, f(3));

  // Shard 4 has origin {0, 0, 4}
  EXPECT_EQ(8, f(4));

  // Shard 5 has origin {0, 2, 4}
  EXPECT_EQ(8, f(5));

  // Shard 6 has origin {0, 0, 6}
  EXPECT_EQ(4, f(6));

  // Shard 7 has origin {0, 2, 6}
  EXPECT_EQ(4, f(7));
}

// Tests the case where shards have varying number of chunks in 2 dimensions.
TEST(GetChunksPerVolumeShardFunctionTest, PartialShards2Dims) {
  ShardingSpec sharding_spec{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  const Index volume_shape[3] = {99, 70, 90};
  const Index chunk_shape[3] = {50, 25, 13};
  // Grid shape: {2, 3, 7}
  // Full shard shape is {2, 2, 2}.

  ShardChunkHierarchy hierarchy;
  ASSERT_TRUE(GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                                     hierarchy));
  EXPECT_THAT(hierarchy.z_index_bits, ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(hierarchy.grid_shape_in_chunks, ::testing::ElementsAre(2, 3, 7));
  EXPECT_THAT(hierarchy.minishard_shape_in_chunks,
              ::testing::ElementsAre(2, 1, 1));
  EXPECT_THAT(hierarchy.shard_shape_in_chunks, ::testing::ElementsAre(2, 2, 2));
  EXPECT_EQ(3, hierarchy.non_shard_bits);
  EXPECT_EQ(3, hierarchy.shard_bits);

  auto f =
      GetChunksPerVolumeShardFunction(sharding_spec, volume_shape, chunk_shape);

  // Shard 0 has origin {0, 0, 0}
  EXPECT_EQ(8, f(0));

  // Shard 1 has origin {0, 2, 0}
  EXPECT_EQ(4, f(1));

  // Shard 2 has origin {0, 0, 2}
  EXPECT_EQ(8, f(2));

  // Shard 3 has origin {0, 2, 2}
  EXPECT_EQ(4, f(3));

  // Shard 4 has origin {0, 0, 4}
  EXPECT_EQ(8, f(4));

  // Shard 5 has origin {0, 2, 4}
  EXPECT_EQ(4, f(5));

  // Shard 6 has origin {0, 0, 6}
  EXPECT_EQ(4, f(6));

  // Shard 7 has origin {0, 2, 6}
  EXPECT_EQ(2, f(7));
}

TEST(GetChunksPerVolumeShardFunctionTest, NotIdentity) {
  ShardingSpec sharding_spec{
      /*.hash_function=*/ShardingSpec::HashFunction::murmurhash3_x86_128,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  const Index volume_shape[3] = {99, 98, 90};
  const Index chunk_shape[3] = {50, 25, 13};

  ShardChunkHierarchy hierarchy;
  EXPECT_FALSE(GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                                      hierarchy));

  EXPECT_FALSE(GetChunksPerVolumeShardFunction(sharding_spec, volume_shape,
                                               chunk_shape));
}

TEST(GetChunksPerVolumeShardFunctionTest, NotEnoughBits) {
  ShardingSpec sharding_spec{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/2,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  const Index volume_shape[3] = {99, 98, 90};
  const Index chunk_shape[3] = {50, 25, 13};

  ShardChunkHierarchy hierarchy;
  EXPECT_FALSE(GetShardChunkHierarchy(sharding_spec, volume_shape, chunk_shape,
                                      hierarchy));

  EXPECT_FALSE(GetChunksPerVolumeShardFunction(sharding_spec, volume_shape,
                                               chunk_shape));
}

TEST(NeuroglancerPrecomputedCodecSpecTest, Merge) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec1,
      CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec2, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"encoding", "raw"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec3, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"encoding", "jpeg"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec4, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"encoding", "jpeg"},
                                        {"jpeg_quality", 50}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec4a, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                         {"encoding", "jpeg"},
                                         {"jpeg_quality", 55}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec5, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"shard_data_encoding", "raw"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec6, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"shard_data_encoding", "gzip"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec7, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"encoding", "raw"},
                                        {"shard_data_encoding", "raw"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto codec8, CodecSpec::FromJson({{"driver", "neuroglancer_precomputed"},
                                        {"encoding", "raw"},
                                        {"shard_data_encoding", "gzip"}}));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec1), ::testing::Optional(codec1));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec2), ::testing::Optional(codec2));
  EXPECT_THAT(CodecSpec::Merge(codec1, codec3), ::testing::Optional(codec3));
  EXPECT_THAT(CodecSpec::Merge(codec3, codec4), ::testing::Optional(codec4));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec5), ::testing::Optional(codec7));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec6), ::testing::Optional(codec8));
  EXPECT_THAT(CodecSpec::Merge(codec2, codec3),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Cannot merge codec spec .* with .*: \"encoding\" mismatch"));
  EXPECT_THAT(CodecSpec::Merge(codec4, codec4a),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge codec spec .* with .*: "
                            "\"jpeg_quality\" mismatch"));
  EXPECT_THAT(CodecSpec::Merge(codec5, codec6),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge codec spec .* with .*: "
                            "\"shard_data_encoding\" mismatch"));
}

TEST(NeuroglancerPrecomputedCodecSpecTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<tensorstore::CodecSpec>({
      ::nlohmann::json::value_t::discarded,
      {
          {"driver", "neuroglancer_precomputed"},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"encoding", "jpeg"},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"encoding", "jpeg"},
          {"jpeg_quality", 80},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"encoding", "raw"},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"encoding", "compressed_segmentation"},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"shard_data_encoding", "gzip"},
      },
      {
          {"driver", "neuroglancer_precomputed"},
          {"shard_data_encoding", "raw"},
      },
  });
}

}  // namespace
