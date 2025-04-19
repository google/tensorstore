// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/driver/tiff/metadata.h"  // Header file being tested

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/internal/json_binding/gtest.h"  // For TestJsonBinderRoundTrip
#include "tensorstore/internal/json_gtest.h"          // For MatchesJson
#include "tensorstore/kvstore/tiff/tiff_details.h"  // For ImageDirectory, enums etc.
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"  // For TiffParseResult
#include "tensorstore/schema.h"
#include "tensorstore/util/status_testutil.h"  // For TENSORSTORE_ASSERT_OK_AND_ASSIGN, MatchesStatus

namespace {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::dtype_v;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_tiff::CreateMetadataFromParseResult;
using ::tensorstore::internal_tiff::TiffMetadata;
using ::tensorstore::internal_tiff::TiffMetadataConstraints;
using ::tensorstore::internal_tiff_kvstore::CompressionType;
using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::PlanarConfigType;
using ::tensorstore::internal_tiff_kvstore::SampleFormatType;
using ::tensorstore::internal_tiff_kvstore::TiffParseResult;
using ::testing::ElementsAre;

// --- Helper functions to create test data ---

// Creates a basic valid ImageDirectory (uint8, 1 sample, chunky, no
// compression, tiled)
ImageDirectory MakeBasicImageDirectory(uint32_t width = 100,
                                       uint32_t height = 80,
                                       uint32_t tile_width = 16,
                                       uint32_t tile_height = 16) {
  ImageDirectory dir;
  dir.width = width;
  dir.height = height;
  dir.tile_width = tile_width;
  dir.tile_height = tile_height;
  dir.rows_per_strip = 0;  // Indicates tiled
  dir.samples_per_pixel = 1;
  dir.compression = static_cast<uint16_t>(CompressionType::kNone);
  dir.photometric = 1;  // BlackIsZero
  dir.planar_config = static_cast<uint16_t>(PlanarConfigType::kChunky);
  dir.bits_per_sample = {8};
  dir.sample_format = {
      static_cast<uint16_t>(SampleFormatType::kUnsignedInteger)};
  // Offsets/bytecounts not needed for CreateMetadataFromParseResult tests
  return dir;
}

// Creates a TiffParseResult containing the given directories
TiffParseResult MakeParseResult(std::vector<ImageDirectory> dirs) {
  TiffParseResult result;
  result.image_directories = std::move(dirs);
  // Other TiffParseResult fields (endian, raw directories) are not used by
  // CreateMetadataFromParseResult, so leave them default.
  return result;
}

// --- Tests for TiffMetadataConstraints ---
TEST(MetadataConstraintsTest, JsonBindingRoundTrip) {
  TiffMetadataConstraints constraints;
  constraints.ifd_index = 5;
  constraints.dtype = dtype_v<tensorstore::dtypes::float32_t>;
  constraints.shape = {{100, 200}};
  constraints.rank = 2;

  ::nlohmann::json expected_json = {
      {"ifd_index", 5}, {"dtype", "float32"}, {"shape", {100, 200}}};

  tensorstore::TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>(
      {expected_json});

  // Test with defaults excluded
  auto expected_json_defaults_excluded = ::nlohmann::json::object();
  tensorstore::TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>(
      {expected_json_defaults_excluded});

  // Test with defaults included
  ::nlohmann::json expected_json_defaults_included = {{"ifd_index", 0}};

  tensorstore::TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>(
      {expected_json_defaults_included}, jb::DefaultBinder<>,
      tensorstore::IncludeDefaults{true});
}

TEST(MetadataConstraintsTest, JsonBindingInvalid) {
  EXPECT_THAT(TiffMetadataConstraints::FromJson({{"ifd_index", "abc"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TiffMetadataConstraints::FromJson({{"dtype", 123}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TiffMetadataConstraints::FromJson({{"shape", {10, "a"}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// --- Tests for CreateMetadataFromParseResult ---
TEST(CreateMetadataTest, BasicSuccessTile) {
  auto parse_result =
      MakeParseResult({MakeBasicImageDirectory(100, 80, 16, 16)});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  const auto& m = *metadata_ptr;

  EXPECT_EQ(m.ifd_index, 0);
  EXPECT_EQ(m.num_ifds, 1);
  EXPECT_EQ(m.rank, 2);
  EXPECT_THAT(m.shape, ElementsAre(80, 100));  // Y, X
  EXPECT_EQ(m.dtype, dtype_v<uint8_t>);
  EXPECT_EQ(m.samples_per_pixel, 1);
  EXPECT_EQ(m.compression_type, CompressionType::kNone);
  EXPECT_EQ(m.planar_config, PlanarConfigType::kChunky);
  EXPECT_THAT(m.chunk_layout.read_chunk().shape(),
              ElementsAre(16, 16));  // TileH, TileW
  EXPECT_THAT(m.chunk_layout.inner_order(),
              ElementsAre(1, 0));  // X faster than Y
  // CodecSpec should be default initialized
  EXPECT_FALSE(m.codec_spec.valid());
}

TEST(CreateMetadataTest, BasicSuccessStrip) {
  ImageDirectory img_dir = MakeBasicImageDirectory(100, 80);
  img_dir.tile_width = 0;  // Indicate strips
  img_dir.tile_height = 0;
  img_dir.rows_per_strip = 10;
  auto parse_result = MakeParseResult({img_dir});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  const auto& m = *metadata_ptr;

  EXPECT_EQ(m.rank, 2);
  EXPECT_THAT(m.shape, ElementsAre(80, 100));
  EXPECT_EQ(m.dtype, dtype_v<uint8_t>);
  EXPECT_THAT(m.chunk_layout.read_chunk().shape(),
              ElementsAre(10, 100));  // RowsPerStrip, Full Width
  EXPECT_THAT(m.chunk_layout.inner_order(), ElementsAre(1, 0));
}

TEST(CreateMetadataTest, MultiSampleChunky) {
  ImageDirectory img_dir = MakeBasicImageDirectory(100, 80, 16, 16);
  img_dir.samples_per_pixel = 3;
  img_dir.bits_per_sample = {8, 8, 8};
  img_dir.sample_format = {1, 1, 1};  // Unsigned Int
  img_dir.planar_config = static_cast<uint16_t>(PlanarConfigType::kChunky);
  auto parse_result = MakeParseResult({img_dir});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  const auto& m = *metadata_ptr;

  EXPECT_EQ(m.rank, 3);
  EXPECT_THAT(m.shape, ElementsAre(80, 100, 3));  // Y, X, C
  EXPECT_EQ(m.dtype, dtype_v<uint8_t>);
  EXPECT_EQ(m.samples_per_pixel, 3);
  EXPECT_EQ(m.planar_config, PlanarConfigType::kChunky);
  EXPECT_THAT(m.chunk_layout.read_chunk().shape(),
              ElementsAre(16, 16, 3));  // TileH, TileW, Samples
  EXPECT_THAT(m.chunk_layout.inner_order(),
              ElementsAre(2, 1, 0));  // C faster than X faster than Y
}

TEST(CreateMetadataTest, Float32) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.bits_per_sample = {32};
  img_dir.sample_format = {static_cast<uint16_t>(SampleFormatType::kIEEEFloat)};
  auto parse_result = MakeParseResult({img_dir});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  EXPECT_EQ(metadata_ptr->dtype, dtype_v<tensorstore::dtypes::float32_t>);
}

TEST(CreateMetadataTest, Int16) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.bits_per_sample = {16};
  img_dir.sample_format = {
      static_cast<uint16_t>(SampleFormatType::kSignedInteger)};
  auto parse_result = MakeParseResult({img_dir});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  EXPECT_EQ(metadata_ptr->dtype, dtype_v<int16_t>);
}

TEST(CreateMetadataTest, InvalidIfdIndex) {
  auto parse_result =
      MakeParseResult({MakeBasicImageDirectory()});  // Only IFD 0 exists
  EXPECT_THAT(
      CreateMetadataFromParseResult(parse_result, 1),
      MatchesStatus(absl::StatusCode::kNotFound, ".*IFD index 1 not found.*"));
}

TEST(CreateMetadataTest, UnsupportedPlanar) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.planar_config = static_cast<uint16_t>(PlanarConfigType::kPlanar);
  auto parse_result = MakeParseResult({img_dir});
  EXPECT_THAT(CreateMetadataFromParseResult(parse_result, 0),
              MatchesStatus(absl::StatusCode::kUnimplemented,
                            ".*PlanarConfiguration=2 is not supported.*"));
}

TEST(CreateMetadataTest, UnsupportedCompression) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.compression =
      static_cast<uint16_t>(CompressionType::kLZW);  // Use LZW
  auto parse_result = MakeParseResult({img_dir});
  EXPECT_THAT(CreateMetadataFromParseResult(parse_result, 0),
              MatchesStatus(absl::StatusCode::kUnimplemented,
                            ".*Compression type 5 is not supported.*"));
}

TEST(CreateMetadataTest, InconsistentSamplesMetadata) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.samples_per_pixel = 3;
  img_dir.bits_per_sample = {8, 16, 8};  // Inconsistent bits
  img_dir.sample_format = {1, 1, 1};
  auto parse_result = MakeParseResult({img_dir});
  EXPECT_THAT(CreateMetadataFromParseResult(parse_result, 0),
              MatchesStatus(absl::StatusCode::kUnimplemented,
                            ".*Varying bits_per_sample.*not yet supported.*"));
}

TEST(CreateMetadataTest, MissingRequiredTag) {
  ImageDirectory img_dir = MakeBasicImageDirectory();
  img_dir.width = 0;  // Simulate missing/invalid width tag parsing
  auto parse_result = MakeParseResult({img_dir});
  // Check if shape derivation fails
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata_ptr, CreateMetadataFromParseResult(parse_result, 0));
  EXPECT_THAT(metadata_ptr->shape,
              ElementsAre(80, 0));  // Shape reflects invalid width

  img_dir = MakeBasicImageDirectory();
  img_dir.bits_per_sample.clear();  // Missing bits per sample
  parse_result = MakeParseResult({img_dir});
  EXPECT_THAT(CreateMetadataFromParseResult(parse_result, 0),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Incomplete TIFF metadata.*"));
}

// --- Tests for ValidateMetadataSchema ---

// Helper to get a basic valid metadata object for validation tests
// Moved before first use
tensorstore::Result<std::shared_ptr<const TiffMetadata>>
GetValidTestMetadata() {
  auto parse_result =
      MakeParseResult({MakeBasicImageDirectory(100, 80, 16, 16)});
  // CreateMetadataFromParseResult only returns basic metadata.
  // We need to simulate the full ResolveMetadata step for a complete object.
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               CreateMetadataFromParseResult(parse_result, 0));
  // Manually finalize layout and set fill value for testing
  // ValidateMetadataSchema
  TENSORSTORE_RETURN_IF_ERROR(metadata->chunk_layout.Finalize());
  metadata->fill_value = tensorstore::AllocateArray(
      metadata->chunk_layout.read_chunk().shape(), tensorstore::c_order,
      tensorstore::value_init, metadata->dtype);
  return std::const_pointer_cast<const TiffMetadata>(metadata);
}

TEST(ValidateSchemaTest, CompatibleSchema) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;

  // Compatible rank
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::RankConstraint{2}));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
  TENSORSTORE_ASSERT_OK(
      schema.Set(tensorstore::RankConstraint{tensorstore::dynamic_rank}));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));

  // Compatible dtype
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<uint8_t>));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::DataType()));

  // Compatible domain
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::IndexDomain({80, 100})));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));

  // Compatible domain (subset)
  {
    tensorstore::Schema schema_subset;
    TENSORSTORE_ASSERT_OK(schema_subset.Set(
        tensorstore::IndexDomain(tensorstore::Box({10, 20}, {30, 40}))));
    TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema_subset));
  }

  // Compatible chunk layout (rank match, other constraints compatible)
  tensorstore::ChunkLayout chunk_layout;
  TENSORSTORE_ASSERT_OK(chunk_layout.Set(tensorstore::RankConstraint{2}));
  TENSORSTORE_ASSERT_OK(schema.Set(chunk_layout));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
  TENSORSTORE_ASSERT_OK(
      chunk_layout.Set(tensorstore::ChunkLayout::ChunkShape({16, 16})));
  TENSORSTORE_ASSERT_OK(schema.Set(chunk_layout));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::ChunkLayout()));
  // Compatible codec (default matches default)
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::CodecSpec()));
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
}

TEST(ValidateSchemaTest, IncompatibleRank) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::RankConstraint{3}));
  EXPECT_THAT(ValidateMetadataSchema(*metadata, schema),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Rank.*3.*does not match.*2.*"));
}

TEST(ValidateSchemaTest, IncompatibleDtype) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<tensorstore::dtypes::float32_t>));
  EXPECT_THAT(ValidateMetadataSchema(*metadata, schema),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*dtype.*uint8.*does not match.*float32.*"));
}

TEST(ValidateSchemaTest, IncompatibleDomain) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::IndexDomain({80, 101})));
  EXPECT_THAT(
      ValidateMetadataSchema(*metadata, schema),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    ".*Schema domain .* is not contained .* metadata.*"));
}

TEST(ValidateSchemaTest, IncompatibleChunkLayout) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;
  tensorstore::ChunkLayout chunk_layout;

  chunk_layout = tensorstore::ChunkLayout();
  TENSORSTORE_ASSERT_OK(chunk_layout.Set(tensorstore::RankConstraint{2}));
  TENSORSTORE_ASSERT_OK(
      chunk_layout.Set(tensorstore::ChunkLayout::InnerOrder({0, 1})));
  TENSORSTORE_ASSERT_OK(schema.Set(chunk_layout));
  // This check might pass if MergeFrom succeeded in ResolveMetadata
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));

  chunk_layout = tensorstore::ChunkLayout();
  TENSORSTORE_ASSERT_OK(chunk_layout.Set(tensorstore::RankConstraint{2}));
  TENSORSTORE_ASSERT_OK(
      chunk_layout.Set(tensorstore::ChunkLayout::ChunkShape({32, 32})));
  TENSORSTORE_ASSERT_OK(schema.Set(chunk_layout));
  // This check might also pass if MergeFrom adapted. Validation is primarily
  // during merge.
  TENSORSTORE_EXPECT_OK(ValidateMetadataSchema(*metadata, schema));
}

TEST(ValidateSchemaTest, IncompatibleFillValue) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, GetValidTestMetadata());
  tensorstore::Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(tensorstore::Schema::FillValue(
      tensorstore::MakeArray<uint8_t>({10}))));  // Different value
  EXPECT_THAT(ValidateMetadataSchema(*metadata, schema),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*fill_value.*not supported.*"));
}

}  // namespace