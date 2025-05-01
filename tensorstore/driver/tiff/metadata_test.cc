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

#include "tensorstore/driver/tiff/metadata.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/tiff/compressor.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"
#include "tensorstore/schema.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::AllocateArray;
using ::tensorstore::Box;
using ::tensorstore::ChunkLayout;
using ::tensorstore::CodecSpec;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::DataType;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dtype_v;
using ::tensorstore::dynamic_rank;
using ::tensorstore::endian;
using ::tensorstore::GetConstantVector;
using ::tensorstore::Index;
using ::tensorstore::IndexDomain;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::RankConstraint;
using ::tensorstore::Result;
using ::tensorstore::Schema;
using ::tensorstore::SharedArray;
using ::tensorstore::SharedArrayView;
using ::tensorstore::span;
using ::tensorstore::TestJsonBinderRoundTrip;
using ::tensorstore::TestJsonBinderRoundTripJsonOnly;
using ::tensorstore::internal::CodecDriverSpec;
using ::tensorstore::internal_tiff::Compressor;
using ::tensorstore::internal_tiff::TiffCodecSpec;
using ::tensorstore::internal_tiff::TiffMetadata;
using ::tensorstore::internal_tiff::TiffMetadataConstraints;
using ::tensorstore::internal_tiff::TiffSpecOptions;
using ::tensorstore::internal_tiff_kvstore::CompressionType;
using ::tensorstore::internal_tiff_kvstore::Endian;
using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::PlanarConfigType;
using ::tensorstore::internal_tiff_kvstore::SampleFormatType;
using ::tensorstore::internal_tiff_kvstore::TiffParseResult;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Optional;

// --- Helper functions to create test data ---

// Helper to calculate the number of chunks/tiles/strips
std::tuple<uint64_t, uint32_t, uint32_t> CalculateChunkCounts(
    uint32_t image_width, uint32_t image_height, uint32_t chunk_width,
    uint32_t chunk_height) {
  if (chunk_width == 0 || chunk_height == 0) {
    return {0, 0, 0};
  }
  uint32_t num_cols = (image_width + chunk_width - 1) / chunk_width;
  uint32_t num_rows = (image_height + chunk_height - 1) / chunk_height;
  uint64_t num_chunks = static_cast<uint64_t>(num_rows) * num_cols;
  return {num_chunks, num_rows, num_cols};
}

// Creates a basic valid ImageDirectory.
ImageDirectory MakeImageDirectory(
    uint32_t width = 100, uint32_t height = 80, uint32_t chunk_width = 16,
    uint32_t chunk_height = 16, bool is_tiled = true,
    uint16_t samples_per_pixel = 1, uint16_t bits_per_sample = 8,
    SampleFormatType sample_format = SampleFormatType::kUnsignedInteger,
    CompressionType compression = CompressionType::kNone,
    PlanarConfigType planar_config = PlanarConfigType::kChunky) {
  ImageDirectory dir;
  dir.width = width;
  dir.height = height;
  dir.is_tiled = is_tiled;
  if (is_tiled) {
    dir.chunk_width = chunk_width;
    dir.chunk_height = chunk_height;
  } else {
    dir.chunk_width = width;
    dir.chunk_height = chunk_height;
  }
  dir.samples_per_pixel = samples_per_pixel;
  dir.compression = static_cast<uint16_t>(compression);
  dir.photometric = (samples_per_pixel >= 3) ? 2 : 1;
  dir.planar_config = static_cast<uint16_t>(planar_config);
  dir.bits_per_sample.assign(samples_per_pixel, bits_per_sample);
  dir.sample_format.assign(samples_per_pixel,
                           static_cast<uint16_t>(sample_format));

  // Calculate number of chunks and populate dummy offset/counts
  uint64_t num_chunks;
  uint32_t num_rows, num_cols;
  std::tie(num_chunks, num_rows, num_cols) = CalculateChunkCounts(
      dir.width, dir.height, dir.chunk_width, dir.chunk_height);

  // For planar, the count is per plane
  if (planar_config == PlanarConfigType::kPlanar && samples_per_pixel > 1) {
    num_chunks *= samples_per_pixel;
  }

  // Dummy offset and size.
  dir.chunk_offsets.assign(num_chunks, 1000);
  dir.chunk_bytecounts.assign(
      num_chunks, dir.chunk_width * dir.chunk_height * bits_per_sample / 8);

  return dir;
}

// Creates a TiffParseResult containing the given directories
TiffParseResult MakeParseResult(std::vector<ImageDirectory> dirs,
                                Endian endian = Endian::kLittle) {
  TiffParseResult result;
  result.image_directories = std::move(dirs);
  result.endian = endian;
  result.full_read = true;  // Assume fully parsed for tests
  // Other TiffParseResult fields not used by ResolveMetadata yet.
  return result;
}
// --- Tests for TiffSpecOptions ---
TEST(SpecOptionsTest, JsonBindingDefault) {
  // Default is single IFD 0
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>(
      {
          /*expected_json=*/{{"ifd", 0}},
      },
      jb::DefaultBinder<>, tensorstore::IncludeDefaults{true});
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>(
      {
          /*expected_json=*/::nlohmann::json::object(),
      },
      jb::DefaultBinder<>, tensorstore::IncludeDefaults{false});
}

TEST(SpecOptionsTest, JsonBindingSingleIfdExplicit) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd", 5}},  // Explicit IFD
  });
}

TEST(SpecOptionsTest, JsonBindingStackingSimple) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking", {{"dimensions", {"z"}}, {"ifd_count", 10}}}},
  });
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking", {{"dimensions", {"z"}}, {"dimension_sizes", {10}}}}},
  });
}

TEST(SpecOptionsTest, JsonBindingStackingMultiDim) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking",
        {{"dimensions", {"t", "c"}}, {"dimension_sizes", {5, 3}}}}},
  });
}

TEST(SpecOptionsTest, JsonBindingStackingMultiDimWithCount) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking",
        {{"dimensions", {"t", "c"}},
         {"dimension_sizes", {5, 3}},
         {"ifd_count", 15}}}},
  });
}

TEST(SpecOptionsTest, JsonBindingStackingWithSequenceOrder) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking",
        {{"dimensions", {"t", "c"}},
         {"dimension_sizes", {5, 3}},
         {"ifd_sequence_order", {"c", "t"}}}}},
  });
}

TEST(SpecOptionsTest, JsonBindingWithSampleLabel) {
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd", 3}, {"sample_dimension_label", "channel"}},
  });
  TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
      {{"ifd_stacking", {{"dimensions", {"z"}}, {"ifd_count", 10}}},
       {"sample_dimension_label", "rgba"}},
  });
}

TEST(SpecOptionsTest, JsonBindingInvalidIfdNegative) {
  EXPECT_THAT(TiffSpecOptions::FromJson({{"ifd", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingMissingDims) {
  EXPECT_THAT(
      TiffSpecOptions::FromJson({{"ifd_stacking", {{"ifd_count", 10}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"dimensions\".*missing.*"));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingEmptyDims) {
  EXPECT_THAT(
      TiffSpecOptions::FromJson(
          {{"ifd_stacking",
            {{"dimensions", nlohmann::json::array()}, {"ifd_count", 10}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"dimensions\" must not be empty.*"));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingSizeMismatch) {
  // dim_sizes length mismatch
  EXPECT_THAT(TiffSpecOptions::FromJson(
                  {{"ifd_stacking",
                    {{"dimensions", {"t", "c"}}, {"dimension_sizes", {5}}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"dimension_sizes\" length \\(1\\) must match "
                            "\"dimensions\" length \\(2\\).*"));  // KEEP
  // ifd_count mismatch with dim_sizes product
  EXPECT_THAT(
      TiffSpecOptions::FromJson({{"ifd_stacking",
                                  {{"dimensions", {"t", "c"}},
                                   {"dimension_sizes", {5, 3}},
                                   {"ifd_count", 16}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*Product of \"dimension_sizes\" \\(15\\) does not "
                    "match specified \"ifd_count\" \\(16\\).*"));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingMissingSizeInfo) {
  // Rank 1 stack needs either dimension_sizes or ifd_count
  EXPECT_THAT(
      TiffSpecOptions::FromJson({{"ifd_stacking", {{"dimensions", {"z"}}}}}),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*Either \"dimension_sizes\" or \"ifd_count\" must be specified.*"));
  // Rank > 1 stack needs dimension_sizes
  EXPECT_THAT(
      TiffSpecOptions::FromJson(
          {{"ifd_stacking", {{"dimensions", {"z", "t"}}, {"ifd_count", 10}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"dimension_sizes\" must be specified when.*"));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingSequenceOrder) {
  // Sequence order wrong length
  EXPECT_THAT(
      TiffSpecOptions::FromJson({{"ifd_stacking",
                                  {{"dimensions", {"t", "c"}},
                                   {"dimension_sizes", {5, 3}},
                                   {"ifd_sequence_order", {"t"}}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"ifd_sequence_order\" length \\(1\\) must match "
                    "\"dimensions\" length \\(2\\).*"));
  // Sequence order not a permutation
  EXPECT_THAT(
      TiffSpecOptions::FromJson(
          {{"ifd_stacking",
            {
                {"dimensions", {"t", "c"}},
                {"dimension_sizes", {5, 3}},
                {"ifd_sequence_order", {"t", "z"}}  // "z" not in dimensions
            }}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"ifd_sequence_order\" must be a permutation of "
                    "\"dimensions\".*"));
}

TEST(SpecOptionsTest, JsonBindingInvalidStackingDuplicateDimLabel) {
  EXPECT_THAT(TiffSpecOptions::FromJson({{"ifd_stacking",
                                          {{"dimensions", {"z", "z"}},
                                           {"dimension_sizes", {5, 3}}}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Duplicate dimension label \"z\".*"));
}

// --- Tests for TiffMetadataConstraints ---
TEST(MetadataConstraintsTest, JsonBinding) {
  TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>({
      ::nlohmann::json::object(),  // Empty constraints
      {{"dtype", "float32"}},
      {{"shape", {100, 200}}},
      {{"dtype", "int16"}, {"shape", {50, 60, 70}}},
  });

  EXPECT_THAT(TiffMetadataConstraints::FromJson({{"dtype", 123}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(TiffMetadataConstraints::FromJson({{"shape", {10, "a"}}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// --- Tests for TiffCodecSpec ---
TEST(TiffCodecSpecJsonTest, RoundTrip) {
  const std::vector<std::pair<TiffCodecSpec, ::nlohmann::json>> cases = {
      // Test empty/default (unconstrained)
      {{}, ::nlohmann::json::object()},
      // Test raw
      {[] {
         TiffCodecSpec spec;
         spec.compression_type = CompressionType::kNone;
         return spec;
       }(),
       {{"compression", "raw"}}},
      // Test LZW
      {[] {
         TiffCodecSpec spec;
         spec.compression_type = CompressionType::kLZW;
         return spec;
       }(),
       {{"compression", "lzw"}}},
      // Test Deflate
      {[] {
         TiffCodecSpec spec;
         spec.compression_type = CompressionType::kDeflate;
         return spec;
       }(),
       {{"compression", "deflate"}}},
      // Add other compression types here as needed
  };

  for (auto& [value, expected_json] : cases) {
    // Test ToJson (CANT GET THIS TO BUILD. TODO: FIX)
    // EXPECT_THAT(jb::ToJson(value),
    // ::testing::Optional(tensorstore::MatchesJson(expected_json)));
    // Test FromJson
    EXPECT_THAT(TiffCodecSpec::FromJson(expected_json),
                ::testing::Optional(value));
  }

  // Test invalid string
  EXPECT_THAT(
      TiffCodecSpec::FromJson({{"compression", "invalid"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*Expected one of .* but received: \"invalid\".*"));
  // Test invalid type
  EXPECT_THAT(TiffCodecSpec::FromJson({{"compression", 123}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Expected one of .* but received: 123.*"));
}

TEST(TiffCodecSpecMergeTest, Merging) {
  // Create heap-allocated objects managed by IntrusivePtr (like CodecSpec does)
  auto ptr_lzw = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_lzw->compression_type = CompressionType::kLZW;

  auto ptr_deflate = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_deflate->compression_type = CompressionType::kDeflate;

  auto ptr_empty = CodecDriverSpec::Make<TiffCodecSpec>();

  auto ptr_none = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_none->compression_type = CompressionType::kNone;

  // --- Test merging INTO spec_lzw ---
  TiffCodecSpec target;
  target.compression_type = CompressionType::kLZW;

  TiffCodecSpec target_copy = target;
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_empty));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));

  target_copy = target;
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_lzw));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));

  target_copy = target;
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_none));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));

  // Test the failing case
  target_copy = target;
  // Call DoMergeFrom directly
  absl::Status merge_status = target_copy.DoMergeFrom(*ptr_deflate);
  ASSERT_FALSE(merge_status.ok());
  EXPECT_EQ(merge_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(merge_status.message(),
              ::testing::HasSubstr("TIFF compression type mismatch"));

  // --- Test merging INTO spec_empty ---
  target_copy = TiffCodecSpec{};  // Empty target
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_lzw));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));

  // --- Test merging INTO spec_none ---
  target_copy = TiffCodecSpec{};  // None target
  target_copy.compression_type = CompressionType::kNone;
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_lzw));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));
}

// --- Tests for ResolveMetadata ---

// Helper to check basic metadata properties
void CheckBaseMetadata(
    const TiffMetadata& md, uint32_t expected_ifd, uint32_t expected_num_ifds,
    DimensionIndex expected_rank, const std::vector<Index>& expected_shape,
    DataType expected_dtype, uint16_t expected_spp,
    CompressionType expected_comp, PlanarConfigType expected_planar,
    const std::vector<Index>& expected_read_chunk_shape,
    const std::vector<DimensionIndex>& expected_inner_order) {
  EXPECT_EQ(md.base_ifd_index, expected_ifd);
  EXPECT_EQ(md.num_ifds_read, expected_num_ifds);
  EXPECT_EQ(md.rank, expected_rank);
  EXPECT_THAT(md.shape, ElementsAreArray(expected_shape));
  EXPECT_EQ(md.dtype, expected_dtype);
  EXPECT_EQ(md.samples_per_pixel, expected_spp);
  EXPECT_EQ(md.compression_type, expected_comp);
  EXPECT_EQ(md.planar_config, expected_planar);
  EXPECT_THAT(md.chunk_layout.read_chunk_shape(),
              ElementsAreArray(expected_read_chunk_shape));
  EXPECT_THAT(md.chunk_layout.inner_order(),
              ElementsAreArray(expected_inner_order));
  // Basic check on dimension mapping size
  EXPECT_EQ(md.dimension_mapping.labels_by_ts_dim.size(), expected_rank);
}

TEST(ResolveMetadataTest, BasicSuccessTileChunkySpp1) {
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 1)});
  TiffSpecOptions options;  // ifd_index = 0
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  CheckBaseMetadata(*metadata, 0, 1, 2, {80, 100}, dtype_v<uint8_t>, 1,
                    CompressionType::kNone, PlanarConfigType::kChunky, {16, 16},
                    {0, 1});

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(0));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(1));
  EXPECT_FALSE(metadata->dimension_mapping.ts_sample_dim.has_value());
  EXPECT_TRUE(metadata->dimension_mapping.ts_stacked_dims.empty());
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("y", "x"));
}

TEST(ResolveMetadataTest, BasicSuccessStripChunkySpp1) {
  ImageDirectory img_dir = MakeImageDirectory(100, 80, 0, 10, false, 1);
  auto parse_result = MakeParseResult({img_dir});
  TiffSpecOptions options;
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  CheckBaseMetadata(*metadata, 0, 1, 2, {80, 100}, dtype_v<uint8_t>, 1,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {10, 100}, {0, 1});

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(0));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(1));
}

TEST(ResolveMetadataTest, BasicSuccessTileChunkySpp3) {
  ImageDirectory img_dir = MakeImageDirectory(100, 80, 16, 16, true, 3);
  auto parse_result = MakeParseResult({img_dir});
  TiffSpecOptions options;
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  CheckBaseMetadata(*metadata, 0, 1, 3, {80, 100, 3}, dtype_v<uint8_t>, 3,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {16, 16, 0}, {0, 1, 2});
  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x", "c"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(0));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(1));
  EXPECT_THAT(metadata->dimension_mapping.ts_sample_dim, Optional(2));
  EXPECT_TRUE(metadata->dimension_mapping.ts_stacked_dims.empty());
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("y", "x", "c"));
}

TEST(ResolveMetadataTest, SelectIfd) {
  auto parse_result = MakeParseResult({
      MakeImageDirectory(100, 80, 16, 16, true, 1, 8),  // IFD 0
      MakeImageDirectory(50, 40, 8, 8, true, 3, 16)     // IFD 1
  });
  TiffSpecOptions options;
  options.ifd_index = 1;
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  CheckBaseMetadata(*metadata, 1, 1, 3, {40, 50, 3}, dtype_v<uint16_t>, 3,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {8, 8, 0}, {0, 1, 2});

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x", "c"));
}

TEST(ResolveMetadataTest, InvalidIfdIndex) {
  auto parse_result = MakeParseResult({MakeImageDirectory()});  // Only IFD 0
  TiffSpecOptions options;
  options.ifd_index = 1;
  Schema schema;
  EXPECT_THAT(ResolveMetadata(parse_result, options, schema),
              MatchesStatus(absl::StatusCode::kNotFound,
                            ".*Requested IFD index 1 not found.*"));
}

TEST(ResolveMetadataTest, SchemaMergeChunkShapeConflict) {
  auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16, 16)});
  TiffSpecOptions options;
  Schema schema;
  ChunkLayout schema_layout;
  TENSORSTORE_ASSERT_OK(schema_layout.Set(ChunkLayout::ChunkShape({32, 32})));
  TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));
  EXPECT_THAT(ResolveMetadata(parse_result, options, schema),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*New hard constraint .*16.* does not match "
                            "existing hard constraint .*32.*.*"));
}

TEST(ResolveMetadataTest, SchemaMergeInnerOrder) {
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 1)});
  TiffSpecOptions options;
  Schema schema;
  ChunkLayout schema_layout;
  TENSORSTORE_ASSERT_OK(schema_layout.Set(ChunkLayout::InnerOrder({1, 0})));
  TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  // Schema hard constraint overrides TIFF default soft constraint
  EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(1, 0));
  EXPECT_EQ(metadata->layout_order, ContiguousLayoutOrder::fortran);
  EXPECT_THAT(metadata->chunk_layout.read_chunk_shape(), ElementsAre(16, 16));
}

TEST(ResolveMetadataTest, SchemaOverrideLabels) {
  // Image is 80x100, spp=3 -> initial conceptual order/labels: y, x, c
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 3)});
  TiffSpecOptions options;
  Schema schema;

  // --- FIX START ---
  // Create an IndexDomain with the desired labels and matching rank/shape.
  // The shape needs to match the expected *final* shape deduced from TIFF ({80,
  // 100, 3}). We specify the desired *final* labels here.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto desired_domain,
      IndexDomainBuilder(3)  // Rank 3 (Y, X, C)
          .shape({80, 100, 3})
          .labels({"height", "width", "channel"})  // Set desired final labels
          .Finalize());

  // Set the domain constraint on the schema
  TENSORSTORE_ASSERT_OK(schema.Set(desired_domain));
  // --- FIX END ---

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  // Now check that ResolveMetadata respected the schema's domain labels
  EXPECT_THAT(metadata->dimension_labels,
              ElementsAre("height", "width", "channel"));

  // Check mapping based on conceptual labels ('y', 'x', 'c') matching the
  // *final* labels
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim,
              Optional(0));  // 'y' matched 'height' at index 0
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim,
              Optional(1));  // 'x' matched 'width' at index 1
  EXPECT_THAT(metadata->dimension_mapping.ts_sample_dim,
              Optional(2));  // 'c' matched 'channel' at index 2
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("y", "x", "c"));  // Conceptual order still y,x,c

  // Check that chunk layout inner order reflects the final dimension order
  // The default soft inner order is still {0, 1, 2} relative to the *final*
  // axes
  EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0, 1, 2));
}

// TEST(SpecOptionsTest, JsonBinding) {
//   // Default value
//   TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>(
//       {
//           /*expected_json=*/{{"ifd", 0}},  // Default value is included
//       },
//       jb::DefaultBinder<>, tensorstore::IncludeDefaults{true});

//   // Default value excluded
//   TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>(
//       {
//           /*expected_json=*/::nlohmann::json::object(),
//       },
//       jb::DefaultBinder<>, tensorstore::IncludeDefaults{false});

//   // Explicit value
//   TestJsonBinderRoundTripJsonOnly<TiffSpecOptions>({
//       {{"ifd", 5}},
//   });

//   // Invalid type
//   EXPECT_THAT(TiffSpecOptions::FromJson({{"ifd", "abc"}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument));
//   EXPECT_THAT(
//       TiffSpecOptions::FromJson({{"ifd", -1}}),  // Negative index invalid
//       MatchesStatus(absl::StatusCode::kInvalidArgument));
// }

// TEST(SpecOptionsTest, ManualEmptyObjectRoundTripIncludeDefaults) {
//   ::nlohmann::json input_json = ::nlohmann::json::object();

//   // 1. Test FromJson
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(TiffSpecOptions options_obj,
//                                    TiffSpecOptions::FromJson(input_json));

//   // 2. Verify the parsed object state (should have default value)
//   EXPECT_EQ(options_obj.ifd_index, 0);

//   // 3. Test ToJson with IncludeDefaults{true}
//   ::nlohmann::json expected_json = {{"ifd", 0}};
//   EXPECT_THAT(jb::ToJson(options_obj, jb::DefaultBinder<>,
//                          tensorstore::IncludeDefaults{true}),
//               ::testing::Optional(tensorstore::MatchesJson(expected_json)));
// }

// // --- Tests for TiffMetadataConstraints ---
// TEST(MetadataConstraintsTest, JsonBinding) {
//   // Test empty constraints
//   TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>({
//       /*expected_json=*/::nlohmann::json::object(),
//   });

//   // Test with values
//   TestJsonBinderRoundTripJsonOnly<TiffMetadataConstraints>({
//       {
//           {"dtype", "float32"}, {"shape", {100, 200}}
//           // rank is implicitly derived
//       },
//   });

//   // Test invalid values
//   EXPECT_THAT(TiffMetadataConstraints::FromJson({{"dtype", 123}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument));
//   EXPECT_THAT(TiffMetadataConstraints::FromJson({{"shape", {10, "a"}}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument));
// }

// // --- Tests for TiffCodecSpec ---

// TEST(TiffCodecSpecJsonTest, RoundTrip) {
//   // --- UPDATED: Manual round-trip checks ---
//   const std::vector<std::pair<TiffCodecSpec, ::nlohmann::json>> cases = {
//       // Test empty/default (unconstrained)
//       {{}, ::nlohmann::json::object()},
//       // Test raw
//       {[] {
//          TiffCodecSpec spec;
//          spec.compression_type = CompressionType::kNone;
//          return spec;
//        }(),
//        {{"compression", "raw"}}},
//       // Test LZW
//       {[] {
//          TiffCodecSpec spec;
//          spec.compression_type = CompressionType::kLZW;
//          return spec;
//        }(),
//        {{"compression", "lzw"}}},
//       // Test Deflate
//       {[] {
//          TiffCodecSpec spec;
//          spec.compression_type = CompressionType::kDeflate;
//          return spec;
//        }(),
//        {{"compression", "deflate"}}},
//       // Add other compression types here as needed
//   };

//   for (auto& [value, expected_json] : cases) {
//     // Test ToJson (CANT GET THIS TO BUILD. TODO: FIX)
//     // EXPECT_THAT(jb::ToJson(value),
//     // ::testing::Optional(tensorstore::MatchesJson(expected_json)));
//     // Test FromJson
//     EXPECT_THAT(TiffCodecSpec::FromJson(expected_json),
//                 ::testing::Optional(value));
//   }

//   // Test invalid string
//   EXPECT_THAT(
//       TiffCodecSpec::FromJson({{"compression", "invalid"}}),
//       MatchesStatus(absl::StatusCode::kInvalidArgument,
//                     ".*Expected one of .* but received: \"invalid\".*"));
//   // Test invalid type
//   EXPECT_THAT(TiffCodecSpec::FromJson({{"compression", 123}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*Expected one of .* but received: 123.*"));
// }

// TEST(TiffCompressorBinderTest, Binding) {
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(Compressor compressor_raw,
//                                    Compressor::FromJson({{"type",
//                                    "raw"}}));
//   EXPECT_THAT(compressor_raw, ::testing::IsNull());
//   EXPECT_THAT(Compressor::FromJson({{"type", "lzw"}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*\"lzw\" is not registered.*"));
//   EXPECT_THAT(Compressor::FromJson({{"type", "unknown"}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*\"unknown\" is not registered.*"));
//   EXPECT_THAT(Compressor::FromJson({{"level", 5}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*Error parsing .* \"type\": .* missing.*"));
// }

// // --- Tests for ResolveMetadata ---
// TEST(ResolveMetadataTest, BasicSuccessTile) {
//   auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16,
//   16)}); TiffSpecOptions options;  // ifd_index = 0 Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   EXPECT_EQ(metadata->ifd_index, 0);
//   EXPECT_EQ(metadata->num_ifds, 1);
//   EXPECT_EQ(metadata->rank, 2);
//   EXPECT_THAT(metadata->shape, ElementsAre(80, 100));  // Y, X
//   EXPECT_EQ(metadata->dtype, dtype_v<uint8_t>);
//   EXPECT_EQ(metadata->samples_per_pixel, 1);
//   EXPECT_EQ(metadata->compression_type, CompressionType::kNone);
//   EXPECT_EQ(metadata->planar_config, PlanarConfigType::kChunky);
//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(), ElementsAre(16,
//   16)); EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0,
//   1)); EXPECT_EQ(metadata->compressor, nullptr);
// }

// TEST(ResolveMetadataTest, BasicSuccessStrip) {
//   ImageDirectory img_dir =
//       MakeImageDirectory(100, 80, 0, 0);  // Indicate strips
//   img_dir.rows_per_strip = 10;
//   auto parse_result = MakeParseResult({img_dir});
//   TiffSpecOptions options;
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   EXPECT_EQ(metadata->rank, 2);
//   EXPECT_THAT(metadata->shape, ElementsAre(80, 100));
//   EXPECT_EQ(metadata->dtype, dtype_v<uint8_t>);
//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(),
//               ElementsAre(10, 100));
//   EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0, 1));
// }

// TEST(ResolveMetadataTest, MultiSampleChunky) {
//   ImageDirectory img_dir = MakeImageDirectory(100, 80, 16, 16,
//   /*samples=*/3); auto parse_result = MakeParseResult({img_dir});
//   TiffSpecOptions options;
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   EXPECT_EQ(metadata->rank, 3);
//   EXPECT_THAT(metadata->shape, ElementsAre(80, 100, 3));  // Y, X, C
//   EXPECT_EQ(metadata->dtype, dtype_v<uint8_t>);
//   EXPECT_EQ(metadata->samples_per_pixel, 3);
//   EXPECT_EQ(metadata->planar_config, PlanarConfigType::kChunky);
//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(),
//               ElementsAre(16, 16, 3));
//   EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0, 1, 2));
// }

// TEST(ResolveMetadataTest, SelectIfd) {
//   auto parse_result = MakeParseResult({
//       MakeImageDirectory(100, 80, 16, 16, /*samples=*/1, /*bits=*/8),  //
//       IFD 0 MakeImageDirectory(50, 40, 8, 8, /*samples=*/3, /*bits=*/16) //
//       IFD 1
//   });
//   TiffSpecOptions options;
//   options.ifd_index = 1;  // Select the second IFD
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   EXPECT_EQ(metadata->ifd_index, 1);
//   EXPECT_EQ(metadata->rank, 3);
//   EXPECT_THAT(metadata->shape, ElementsAre(40, 50, 3));  // Y, X, C
//   EXPECT_EQ(metadata->dtype, dtype_v<uint16_t>);
//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(),
//               ElementsAre(8, 8, 3));
// }

// TEST(ResolveMetadataTest, SchemaMergeChunkShape) {
//   auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16,
//   16)}); TiffSpecOptions options; Schema schema; ChunkLayout schema_layout;
//   // Set a chunk shape in the schema that conflicts with the TIFF tile size
//   TENSORSTORE_ASSERT_OK(schema_layout.Set(ChunkLayout::ChunkShape({32,
//   32}))); TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));

//   // Expect an error because the hard constraint from the schema conflicts
//   // with the hard constraint derived from the TIFF tags (16x16).
//   EXPECT_THAT(ResolveMetadata(parse_result, options, schema),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*New hard constraint .*32.* does not match "
//                             "existing hard constraint .*16.*"));
// }

// TEST(ResolveMetadataTest, SchemaMergeChunkShapeCompatible) {
//   // Test merging when the schema chunk shape *matches* the TIFF tile size
//   auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16,
//   16)}); TiffSpecOptions options; Schema schema; ChunkLayout schema_layout;
//   TENSORSTORE_ASSERT_OK(
//       schema_layout.Set(ChunkLayout::ChunkShape({16, 16})));  // Match tile
//       size
//   TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));

//   // This should now succeed
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(), ElementsAre(16,
//   16));
// }

// TEST(ResolveMetadataTest, SchemaMergeInnerOrder) {
//   auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16,
//   16)}); TiffSpecOptions options; Schema schema; ChunkLayout schema_layout;
//   TENSORSTORE_ASSERT_OK(
//       schema_layout.Set(ChunkLayout::InnerOrder({0, 1})));  // Y faster
//       than
//       X
//   TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));

//   // Schema constraint overrides TIFF default inner order
//   EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0, 1));
//   // Chunk shape from TIFF should be retained
//   EXPECT_THAT(metadata->chunk_layout.read_chunk().shape(), ElementsAre(16,
//   16)); EXPECT_THAT(metadata->chunk_layout.grid_origin(),
//               ElementsAre(0, 0));  // Default grid origin
// }

// TEST(ResolveMetadataTest, SchemaCodecCompatible) {
//   auto parse_result = MakeParseResult({MakeImageDirectory()});
//   TiffSpecOptions options;
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto spec,
//       CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "raw"}}));
//   TENSORSTORE_ASSERT_OK(schema.Set(spec));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));
//   EXPECT_EQ(metadata->compression_type, CompressionType::kNone);
//   EXPECT_THAT(metadata->compressor, ::testing::IsNull());
// }
// TEST(ResolveMetadataTest, SchemaCodecIncompatible) {
//   auto parse_result = MakeParseResult({MakeImageDirectory()});
//   TiffSpecOptions options;
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto spec,
//       CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "lzw"}}));
//   TENSORSTORE_ASSERT_OK(schema.Set(spec));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));
// }

// TEST(ResolveMetadataTest, SchemaCodecWrongDriver) {
//   auto parse_result = MakeParseResult({MakeImageDirectory()});
//   TiffSpecOptions options;
//   Schema schema;
//   EXPECT_THAT(CodecSpec::FromJson({{"driver", "n5"}}),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*\"n5\" is not registered.*"));
// }

// TEST(ResolveMetadataTest, SchemaCodecUnspecified) {
//   auto parse_result = MakeParseResult({MakeImageDirectory()});
//   TiffSpecOptions options;
//   Schema schema;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto metadata, ResolveMetadata(parse_result, options, schema));
//   EXPECT_EQ(metadata->compression_type, CompressionType::kNone);
//   EXPECT_THAT(metadata->compressor, ::testing::IsNull());
// }
// TEST(ResolveMetadataTest, UnsupportedCompressionInFile) {
//   ImageDirectory img_dir = MakeImageDirectory();
//   img_dir.compression = static_cast<uint16_t>(CompressionType::kLZW);
//   auto parse_result = MakeParseResult({img_dir});
//   TiffSpecOptions options;
//   Schema schema;
//   EXPECT_THAT(ResolveMetadata(parse_result, options, schema),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*\"lzw\" is not registered.*"));
// }
// TEST(ResolveMetadataTest, InvalidIfdIndex) {
//   auto parse_result = MakeParseResult({MakeImageDirectory()});  // Only IFD
//   0 TiffSpecOptions options; options.ifd_index = 1; Schema schema;
//   EXPECT_THAT(
//       ResolveMetadata(parse_result, options, schema),
//       MatchesStatus(absl::StatusCode::kNotFound, ".*IFD index 1 not
//       found.*"));
// }

// TEST(ResolveMetadataTest, UnsupportedPlanar) {
//   ImageDirectory img_dir = MakeImageDirectory();
//   img_dir.planar_config = static_cast<uint16_t>(PlanarConfigType::kPlanar);
//   auto parse_result = MakeParseResult({img_dir});
//   TiffSpecOptions options;
//   Schema schema;
//   EXPECT_THAT(ResolveMetadata(parse_result, options, schema),
//               MatchesStatus(absl::StatusCode::kUnimplemented,
//                             ".*PlanarConfiguration=2 is not supported.*"));
// }

// // --- Tests for ValidateResolvedMetadata ---

// // Helper to get a basic valid resolved metadata object
// Result<std::shared_ptr<const TiffMetadata>>
// GetResolvedMetadataForValidation() {
//   auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16,
//   16)}); TiffSpecOptions options; Schema schema; return
//   ResolveMetadata(parse_result, options, schema);
// }

// TEST(ValidateResolvedMetadataTest, CompatibleConstraints) {
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
//                                    GetResolvedMetadataForValidation());
//   TiffMetadataConstraints constraints;

//   // No constraints
//   TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));

//   // Matching rank
//   constraints.rank = 2;
//   TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
//   constraints.rank = dynamic_rank;  // Reset

//   // Matching dtype
//   constraints.dtype = dtype_v<uint8_t>;
//   TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
//   constraints.dtype = std::nullopt;  // Reset

//   // Matching shape
//   constraints.shape = {{80, 100}};
//   TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
//   constraints.shape = std::nullopt;  // Reset
// }

// TEST(ValidateResolvedMetadataTest, IncompatibleRank) {
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
//                                    GetResolvedMetadataForValidation());
//   TiffMetadataConstraints constraints;
//   constraints.rank = 3;
//   EXPECT_THAT(
//       ValidateResolvedMetadata(*metadata, constraints),
//       MatchesStatus(
//           absl::StatusCode::kFailedPrecondition,
//           ".*Resolved TIFF rank .*2.* does not match.*constraint rank
//           .*3.*"));
// }

// TEST(ValidateResolvedMetadataTest, IncompatibleDtype) {
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
//                                    GetResolvedMetadataForValidation());
//   TiffMetadataConstraints constraints;
//   constraints.dtype = dtype_v<uint16_t>;
//   EXPECT_THAT(ValidateResolvedMetadata(*metadata, constraints),
//               MatchesStatus(absl::StatusCode::kFailedPrecondition,
//                             ".*Resolved TIFF dtype .*uint8.* does not "
//                             "match.*constraint dtype .*uint16.*"));
// }

// TEST(ValidateResolvedMetadataTest, IncompatibleShape) {
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
//                                    GetResolvedMetadataForValidation());
//   TiffMetadataConstraints constraints;
//   constraints.shape = {{80, 101}};  // Width mismatch
//   EXPECT_THAT(ValidateResolvedMetadata(*metadata, constraints),
//               MatchesStatus(absl::StatusCode::kFailedPrecondition,
//                             ".*Resolved TIFF shape .*80, 100.* does not "
//                             "match.*constraint shape .*80, 101.*"));

//   constraints.shape = {{80}};  // Rank mismatch inferred from shape
//   EXPECT_THAT(ValidateResolvedMetadata(*metadata, constraints),
//               MatchesStatus(absl::StatusCode::kFailedPrecondition,
//                             ".*Rank of resolved TIFF shape .*2.* does not "
//                             "match.*constraint shape .*1.*"));
// }

// // --- Tests for GetEffective... Functions ---

// TEST(GetEffectiveTest, DataType) {
//   TiffMetadataConstraints constraints;
//   Schema schema;

//   // Neither specified -> invalid
//   EXPECT_FALSE(GetEffectiveDataType(constraints, schema).value().valid());

//   // Schema only
//   TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<uint16_t>));
//   EXPECT_THAT(GetEffectiveDataType(constraints, schema),
//               ::testing::Optional(dtype_v<uint16_t>));

//   // Constraints only
//   schema = Schema();
//   constraints.dtype = dtype_v<tensorstore::dtypes::float32_t>;
//   EXPECT_THAT(GetEffectiveDataType(constraints, schema),
//               ::testing::Optional(dtype_v<tensorstore::dtypes::float32_t>));

//   // Both match
//   TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<tensorstore::dtypes::float32_t>));
//   EXPECT_THAT(GetEffectiveDataType(constraints, schema),
//               ::testing::Optional(dtype_v<tensorstore::dtypes::float32_t>));

//   // Both conflict
//   schema = Schema();
//   TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<int32_t>));
//   EXPECT_THAT(
//       GetEffectiveDataType(constraints, schema),
//       MatchesStatus(absl::StatusCode::kInvalidArgument, ".*conflicts.*"));
// }

// TEST(GetEffectiveTest, Domain) {
//   TiffSpecOptions options;
//   TiffMetadataConstraints constraints;
//   Schema schema;

//   // Nothing specified -> unknown domain
//   EXPECT_EQ(IndexDomain<>(),
//             GetEffectiveDomain(options, constraints, schema).value());

//   // Rank from schema
//   TENSORSTORE_ASSERT_OK(schema.Set(RankConstraint{3}));
//   EXPECT_EQ(IndexDomain(3),
//             GetEffectiveDomain(options, constraints, schema).value());

//   // Rank from constraints
//   schema = Schema();
//   constraints.rank = 2;
//   EXPECT_EQ(IndexDomain(2),
//             GetEffectiveDomain(options, constraints, schema).value());

//   // Shape from constraints
//   constraints.shape = {{50, 60}};  // Implies rank 2
//   constraints.rank = dynamic_rank;
//   EXPECT_EQ(IndexDomain({50, 60}),
//             GetEffectiveDomain(options, constraints, schema).value());

//   // Shape from constraints, domain from schema (compatible bounds)
//   schema = Schema();
//   constraints = TiffMetadataConstraints();
//   constraints.shape = {{50, 60}};
//   TENSORSTORE_ASSERT_OK(schema.Set(IndexDomain(Box({0, 0}, {50, 60}))));
//   EXPECT_EQ(IndexDomain(Box({0, 0}, {50, 60})),
//             GetEffectiveDomain(options, constraints, schema).value());

//   // Shape from constraints, domain from schema (incompatible bounds ->
//   Error) schema = Schema(); constraints = TiffMetadataConstraints();
//   constraints.shape = {{50, 60}};
//   TENSORSTORE_ASSERT_OK(
//       schema.Set(IndexDomain(Box({10, 10}, {40, 50}))));  // Origin differs
//   EXPECT_THAT(GetEffectiveDomain(options, constraints, schema),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*Lower bounds do not match.*"));

//   // Shape from constraints, domain from schema (rank incompatible)
//   schema = Schema();
//   constraints = TiffMetadataConstraints();
//   constraints.shape = {{50, 60}};
//   TENSORSTORE_ASSERT_OK(schema.Set(IndexDomain(Box({10}, {40}))));  // Rank
//   1 EXPECT_THAT(
//       GetEffectiveDomain(options, constraints, schema),
//       MatchesStatus(absl::StatusCode::kInvalidArgument,
//       ".*Rank.*conflicts.*"));

//   // Shape from constraints, domain from schema (bounds incompatible)
//   schema = Schema();
//   constraints = TiffMetadataConstraints();
//   constraints.shape = {{30, 40}};
//   TENSORSTORE_ASSERT_OK(schema.Set(
//       IndexDomain(Box({0, 0}, {30, 50}))));  // Dim 1 exceeds constraint
//       shape
//   EXPECT_THAT(GetEffectiveDomain(options, constraints, schema),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*Mismatch in dimension 1.*"));
// }

// TEST(GetEffectiveTest, ChunkLayout) {
//   TiffSpecOptions options;
//   TiffMetadataConstraints constraints;
//   Schema schema;
//   ChunkLayout layout;

//   // Nothing specified -> default layout (rank 0)
//   EXPECT_EQ(ChunkLayout{},
//             GetEffectiveChunkLayout(options, constraints, schema).value());

//   // Rank specified -> default layout for that rank
//   constraints.rank = 2;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       layout, GetEffectiveChunkLayout(options, constraints, schema));
//   EXPECT_EQ(layout.rank(), 2);
//   EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1));
//   EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0));

//   // Schema specifies chunk shape
//   schema = Schema();
//   constraints = TiffMetadataConstraints();
//   constraints.rank = 2;
//   ChunkLayout schema_layout;
//   TENSORSTORE_ASSERT_OK(schema_layout.Set(ChunkLayout::ChunkShape({32,
//   64}))); TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       layout, GetEffectiveChunkLayout(options, constraints, schema));
//   EXPECT_THAT(layout.read_chunk().shape(), ElementsAre(32, 64));
//   EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1));

//   // Schema specifies inner order
//   schema = Schema();
//   constraints = TiffMetadataConstraints();
//   constraints.rank = 2;
//   schema_layout = ChunkLayout();
//   TENSORSTORE_ASSERT_OK(schema_layout.Set(ChunkLayout::InnerOrder({0,
//   1}))); TENSORSTORE_ASSERT_OK(schema.Set(schema_layout));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       layout, GetEffectiveChunkLayout(options, constraints, schema));
//   EXPECT_THAT(layout.inner_order(),
//               ElementsAre(0, 1));  // Schema order overrides default
// }

// TEST(GetEffectiveTest, Codec) {
//   TiffSpecOptions options;
//   TiffMetadataConstraints constraints;
//   Schema schema;
//   CodecDriverSpec::PtrT<TiffCodecSpec> codec_ptr;
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       codec_ptr, GetEffectiveCodec(options, constraints, schema));
//   ASSERT_NE(codec_ptr, nullptr);
//   EXPECT_FALSE(codec_ptr->compression_type.has_value());

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto raw_schema,
//       CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "raw"}}));
//   TENSORSTORE_ASSERT_OK(schema.Set(raw_schema));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       codec_ptr, GetEffectiveCodec(options, constraints, schema));
//   ASSERT_NE(codec_ptr, nullptr);
//   EXPECT_THAT(codec_ptr->compression_type,
//               ::testing::Optional(CompressionType::kNone));

//   schema = Schema();
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto lzw_schema,
//       CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "lzw"}}));
//   TENSORSTORE_ASSERT_OK(schema.Set(lzw_schema));
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       codec_ptr, GetEffectiveCodec(options, constraints, schema));
//   ASSERT_NE(codec_ptr, nullptr);
//   EXPECT_THAT(codec_ptr->compression_type,
//               ::testing::Optional(CompressionType::kLZW));
// }

// // Helper function to encode an array to a Cord for testing DecodeChunk
// Result<absl::Cord> EncodeArrayToCord(SharedArrayView<const void> array,
//                                      tensorstore::endian source_endian,
//                                      ContiguousLayoutOrder order) {
//   absl::Cord cord;
//   riegeli::CordWriter<> writer(&cord);
//   if (!tensorstore::internal::EncodeArrayEndian(array, source_endian,
//   order,
//                                                 writer)) {
//     return writer.status();
//   }
//   if (!writer.Close()) {
//     return writer.status();
//   }
//   return cord;
// }

// // Test fixture for DecodeChunk tests
// class DecodeChunkTest : public ::testing::Test {
//  protected:
//   // Helper to create metadata for testing
//   TiffMetadata CreateMetadata(
//       DataType dtype, span<const Index> shape, span<const Index>
//       chunk_shape, ContiguousLayoutOrder layout_order =
//       ContiguousLayoutOrder::c, Endian endian = Endian::kLittle,
//       CompressionType compression = CompressionType::kNone) {
//     TiffMetadata metadata;
//     metadata.dtype = dtype;
//     metadata.rank = shape.size();
//     metadata.shape.assign(shape.begin(), shape.end());
//     metadata.endian = endian;
//     metadata.compression_type = compression;
//     // metadata.compressor = nullptr;  // Assume no compressor for now

//     // Set chunk layout properties
//     TENSORSTORE_CHECK_OK(
//         metadata.chunk_layout.Set(RankConstraint{metadata.rank}));
//     TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(
//         ChunkLayout::ChunkShape(chunk_shape, /*hard=*/true)));
//     TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(ChunkLayout::GridOrigin(
//         GetConstantVector<Index, 0>(metadata.rank), /*hard=*/true)));
//     std::vector<DimensionIndex> inner_order(metadata.rank);
//     tensorstore::SetPermutation(layout_order, span(inner_order));
//     TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(
//         ChunkLayout::InnerOrder(inner_order, /*hard=*/true)));
//     TENSORSTORE_CHECK_OK(metadata.chunk_layout.Finalize());

//     // Set the resolved layout enum based on the finalized order
//     metadata.layout_order = layout_order;

//     return metadata;
//   }
// };

// TEST_F(DecodeChunkTest, UncompressedUint8CorderLittleEndian) {
//   const Index shape[] = {2, 3};
//   auto metadata = CreateMetadata(dtype_v<uint8_t>, shape, shape,
//                                  ContiguousLayoutOrder::c,
//                                  Endian::kLittle);
//   auto expected_array = MakeArray<uint8_t>({{1, 2, 3}, {4, 5, 6}});
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord, EncodeArrayToCord(expected_array, endian::little,
//                                          ContiguousLayoutOrder::c));

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
//                                    DecodeChunk(metadata, input_cord));
//   SharedArray<const uint8_t> decoded_array(
//       std::static_pointer_cast<const
//       uint8_t>(decoded_array_void.pointer()), expected_array.layout());
//   EXPECT_EQ(decoded_array, expected_array);
// }

// TEST_F(DecodeChunkTest, UncompressedUint16FortranOrderBigEndian) {
//   const Index shape[] = {2, 3};
//   auto metadata = CreateMetadata(dtype_v<uint16_t>, shape, shape,
//                                  ContiguousLayoutOrder::fortran,
//                                  Endian::kBig);
//   auto expected_array = tensorstore::MakeCopy(
//       MakeArray<uint16_t>({{100, 200, 300}, {400, 500, 600}}),
//       ContiguousLayoutOrder::fortran);
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord, EncodeArrayToCord(expected_array, endian::big,
//                                          ContiguousLayoutOrder::fortran));

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
//                                    DecodeChunk(metadata, input_cord));
//   SharedArray<const uint16_t> decoded_array(
//       std::static_pointer_cast<const
//       uint16_t>(decoded_array_void.pointer()), expected_array.layout());

//   EXPECT_EQ(decoded_array, expected_array);
// }

// TEST_F(DecodeChunkTest, UncompressedFloat32CorderBigEndianToNative) {
//   const Index shape[] = {2, 2};
//   // Native endian might be little, source is big
//   auto metadata = CreateMetadata(dtype_v<float>, shape, shape,
//                                  ContiguousLayoutOrder::c, Endian::kBig);
//   auto expected_array = MakeArray<float>({{1.0f, 2.5f}, {-3.0f, 4.75f}});
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord,
//       EncodeArrayToCord(expected_array, endian::big,
//       ContiguousLayoutOrder::c));

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
//                                    DecodeChunk(metadata, input_cord));
//   // Cast the void result to the expected type, preserving layout
//   SharedArray<const float> decoded_array(
//       std::static_pointer_cast<const float>(decoded_array_void.pointer()),
//       expected_array.layout());

//   EXPECT_EQ(decoded_array, expected_array);
// }

// TEST_F(DecodeChunkTest, UncompressedRank3) {
//   const Index shape[] = {2, 3, 2};  // Y, X, C
//   auto metadata = CreateMetadata(dtype_v<int16_t>, shape, shape,
//                                  ContiguousLayoutOrder::c,
//                                  Endian::kLittle);
//   auto expected_array = MakeArray<int16_t>(
//       {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord, EncodeArrayToCord(expected_array, endian::little,
//                                          ContiguousLayoutOrder::c));

//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
//                                    DecodeChunk(metadata, input_cord));
//   // Cast the void result to the expected type, preserving layout
//   SharedArray<const int16_t> decoded_array(
//       std::static_pointer_cast<const
//       int16_t>(decoded_array_void.pointer()), expected_array.layout());

//   EXPECT_EQ(decoded_array, expected_array);
// }

// TEST_F(DecodeChunkTest, ErrorInputTooSmall) {
//   const Index shape[] = {2, 3};
//   auto metadata = CreateMetadata(dtype_v<uint16_t>, shape, shape,
//                                  ContiguousLayoutOrder::c,
//                                  Endian::kLittle);
//   auto expected_array = MakeArray<uint16_t>({{1, 2, 3}, {4, 5, 6}});
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord, EncodeArrayToCord(expected_array, endian::little,
//                                          ContiguousLayoutOrder::c));

//   // Truncate the cord
//   absl::Cord truncated_cord = input_cord.Subcord(0, input_cord.size() - 1);

//   EXPECT_THAT(
//       DecodeChunk(metadata, truncated_cord),
//       MatchesStatus(absl::StatusCode::kInvalidArgument, ".*Not enough
//       data.*"));
// }

// TEST_F(DecodeChunkTest, ErrorExcessData) {
//   const Index shape[] = {2, 3};
//   auto metadata = CreateMetadata(dtype_v<uint8_t>, shape, shape,
//                                  ContiguousLayoutOrder::c,
//                                  Endian::kLittle);
//   auto expected_array = MakeArray<uint8_t>({{1, 2, 3}, {4, 5, 6}});
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       auto input_cord, EncodeArrayToCord(expected_array, endian::little,
//                                          ContiguousLayoutOrder::c));

//   // Add extra data
//   input_cord.Append("extra");

//   EXPECT_THAT(DecodeChunk(metadata, input_cord),
//               MatchesStatus(absl::StatusCode::kInvalidArgument,
//                             ".*End of data expected.*"));
// }

// // --- Placeholder Tests for Compression ---
// // These require compressor implementations to be registered and
// potentially
// // pre-compressed "golden" data.
// TEST_F(DecodeChunkTest, DISABLED_CompressedDeflate) {
//   // 1. Register Deflate compressor (implementation needed separately)
//   //    RegisterTiffCompressor<DeflateCompressor>("deflate", ...);

//   // 2. Create metadata with deflate compression
//   const Index shape[] = {4, 5};
//   auto metadata =
//       CreateMetadata(dtype_v<uint16_t>, shape, shape,
//       ContiguousLayoutOrder::c,
//                      Endian::kLittle, CompressionType::kDeflate);
//   // Get compressor instance via ResolveMetadata or manually for test
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(
//       metadata.compressor,
//       Compressor::FromJson({{"type", "deflate"}}));  // Assumes
//       registration

//   // 3. Create expected *decoded* array
//   auto expected_array = AllocateArray<uint16_t>(shape,
//   ContiguousLayoutOrder::c,
//                                                 tensorstore::value_init);
//   // Fill with some data...
//   for (Index i = 0; i < 4; ++i)
//     for (Index j = 0; j < 5; ++j) expected_array(i, j) = i * 10 + j;

//   // 4. Create *compressed* input cord (requires deflate implementation or
//   // golden data) Example using golden data (replace hex string with actual
//   // compressed bytes) std::string compressed_hex = "789c...";
//   // TENSORSTORE_ASSERT_OK_AND_ASSIGN(absl::Cord input_cord,
//   // HexToCord(compressed_hex));
//   absl::Cord input_cord;  // Placeholder - needs real compressed data
//   GTEST_SKIP()
//       << "Skipping compressed test until compressor impl/data is
//       available.";

//   // 5. Call DecodeChunk and verify
//   TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
//                                    DecodeChunk(metadata, input_cord));
//   // Cast the void result to the expected type, preserving layout
//   SharedArray<const uint16_t> decoded_array(
//       std::static_pointer_cast<const
//       uint16_t>(decoded_array_void.pointer()), expected_array.layout());

//   EXPECT_EQ(decoded_array, expected_array);
// }

}  // namespace