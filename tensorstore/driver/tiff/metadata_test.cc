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
using ::tensorstore::DimensionSet;
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
using ::tensorstore::internal_tiff::GetEffectiveChunkLayout;
using ::tensorstore::internal_tiff::GetEffectiveCompressor;
using ::tensorstore::internal_tiff::GetEffectiveDimensionUnits;
using ::tensorstore::internal_tiff::GetEffectiveDomain;
using ::tensorstore::internal_tiff::GetInitialChunkLayout;
using ::tensorstore::internal_tiff::ResolveMetadata;
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

  uint64_t num_chunks;
  uint32_t num_rows, num_cols;
  std::tie(num_chunks, num_rows, num_cols) = CalculateChunkCounts(
      dir.width, dir.height, dir.chunk_width, dir.chunk_height);

  // For planar, the count is per plane
  if (planar_config == PlanarConfigType::kPlanar && samples_per_pixel > 1) {
    num_chunks *= samples_per_pixel;
  }

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
                            "\"dimensions\" length \\(2\\).*"));
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
  auto ptr_lzw = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_lzw->compression_type = CompressionType::kLZW;

  auto ptr_deflate = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_deflate->compression_type = CompressionType::kDeflate;

  auto ptr_empty = CodecDriverSpec::Make<TiffCodecSpec>();

  auto ptr_none = CodecDriverSpec::Make<TiffCodecSpec>();
  ptr_none->compression_type = CompressionType::kNone;

  // Test merging INTO spec_lzw
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
  absl::Status merge_status = target_copy.DoMergeFrom(*ptr_deflate);
  ASSERT_FALSE(merge_status.ok());
  EXPECT_EQ(merge_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(merge_status.message(),
              ::testing::HasSubstr("TIFF compression type mismatch"));

  // Test merging inro spec_empty
  target_copy = TiffCodecSpec{};
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_lzw));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));

  // Test merging INTO spec_none---
  target_copy = TiffCodecSpec{};
  target_copy.compression_type = CompressionType::kNone;
  TENSORSTORE_EXPECT_OK(target_copy.DoMergeFrom(*ptr_lzw));
  EXPECT_THAT(target_copy.compression_type,
              ::testing::Optional(CompressionType::kLZW));
}

// --- Tests for GetInitialChunkLayout ---
TEST(GetInitialChunkLayoutTest, TiledChunkySpp1) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8,
                         /*is_tiled=*/true, /*spp=*/1);
  DimensionIndex initial_rank = 2;
  std::vector<std::string> initial_labels = {"y", "x"};
  std::string sample_label = "c";

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kChunky, 1, sample_label));

  EXPECT_EQ(layout.rank(), 2);
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);

  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(8, 16));  // {y, x} order
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(8, 16));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(8, 16));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1));  // Default C
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

TEST(GetInitialChunkLayoutTest, StrippedChunkySpp1) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/50, /*height=*/35,
                         /*chunk_width=*/0, /*chunk_height=*/10,
                         /*is_tiled=*/false, /*spp=*/1);
  DimensionIndex initial_rank = 2;
  std::vector<std::string> initial_labels = {"y", "x"};
  std::string sample_label = "c";
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kChunky, 1, sample_label));

  EXPECT_EQ(layout.rank(), 2);
  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(10, 50));
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(10, 50));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(10, 50));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1));
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

TEST(GetInitialChunkLayoutTest, TiledChunkySpp3) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8,
                         /*is_tiled=*/true, /*spp=*/3);
  DimensionIndex initial_rank = 3;
  std::vector<std::string> initial_labels = {"y", "x", "c"};
  std::string sample_label = "c";
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kChunky, 3, sample_label));

  EXPECT_EQ(layout.rank(), 3);
  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(8, 16, 3));
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(8, 16, 3));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(8, 16, 3));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1, 2));
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

TEST(GetInitialChunkLayoutTest, TiledChunkySpp3YXOrder) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8,
                         /*is_tiled=*/true, /*spp=*/3);
  DimensionIndex initial_rank = 3;
  std::vector<std::string> initial_labels = {"c", "y", "x"};
  std::string sample_label = "c";
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kChunky, 3, sample_label));

  EXPECT_EQ(layout.rank(), 3);
  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(3, 8, 16));
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(3, 8, 16));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(3, 8, 16));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1, 2));
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

TEST(GetInitialChunkLayoutTest, TiledPlanarSpp3) {
  ImageDirectory ifd = MakeImageDirectory(
      /*width=*/60, /*height=*/40,
      /*chunk_width=*/16, /*chunk_height=*/8,
      /*is_tiled=*/true, /*spp=*/3,
      /*bits=*/8, SampleFormatType::kUnsignedInteger, CompressionType::kNone,
      /*planar=*/PlanarConfigType::kPlanar);
  DimensionIndex initial_rank = 3;
  std::vector<std::string> initial_labels = {"c", "y", "x"};
  std::string sample_label = "c";
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kPlanar, 3, sample_label));

  EXPECT_EQ(layout.rank(), 3);
  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1, 2));
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

TEST(GetInitialChunkLayoutTest, StackedTiledChunkySpp1) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8,
                         /*is_tiled=*/true, /*spp=*/1);
  DimensionIndex initial_rank = 3;
  std::vector<std::string> initial_labels = {"z", "y", "x"};
  std::string sample_label = "c";
  auto expected_hard_constraints = DimensionSet::UpTo(initial_rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout layout,
      GetInitialChunkLayout(ifd, initial_rank, initial_labels,
                            PlanarConfigType::kChunky, 1, sample_label));

  EXPECT_EQ(layout.rank(), 3);
  EXPECT_THAT(layout.grid_origin(), ElementsAre(0, 0, 0));
  EXPECT_EQ(layout.grid_origin().hard_constraint, expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.read_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.write_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.write_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(span<const Index>(layout.codec_chunk_shape()),
              ElementsAre(1, 8, 16));
  EXPECT_EQ(layout.codec_chunk_shape().hard_constraint,
            expected_hard_constraints);

  EXPECT_THAT(layout.inner_order(), ElementsAre(0, 1, 2));
  EXPECT_FALSE(layout.inner_order().hard_constraint);
}

// --- Tests for GetEffectiveChunkLayout ---
TEST(GetEffectiveChunkLayoutTest, InitialOnly) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8);
  DimensionIndex rank = 2;
  std::vector<std::string> labels = {"y", "x"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout initial_layout,
      GetInitialChunkLayout(ifd, rank, labels, PlanarConfigType::kChunky, 1,
                            "c"));
  Schema schema;
  DimensionSet expected_hard_constraints = DimensionSet::UpTo(rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout effective_layout,
      GetEffectiveChunkLayout(initial_layout, schema));

  EXPECT_EQ(effective_layout.rank(), 2);
  EXPECT_THAT(span<const Index>(effective_layout.read_chunk_shape()),
              ElementsAre(8, 16));
  EXPECT_EQ(effective_layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.grid_origin(), ElementsAre(0, 0));
  EXPECT_EQ(effective_layout.grid_origin().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.inner_order(), ElementsAre(0, 1));
  EXPECT_EQ(effective_layout.inner_order().hard_constraint,
            initial_layout.inner_order().hard_constraint);
}

TEST(GetEffectiveChunkLayoutTest, SchemaHardInnerOrder) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8);
  DimensionIndex rank = 2;
  std::vector<std::string> labels = {"y", "x"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout initial_layout,
      GetInitialChunkLayout(ifd, rank, labels, PlanarConfigType::kChunky, 1,
                            "c"));
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(ChunkLayout::InnerOrder({1, 0}, /*hard=*/true)));
  DimensionSet expected_hard_constraints = DimensionSet::UpTo(rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout effective_layout,
      GetEffectiveChunkLayout(initial_layout, schema));

  EXPECT_THAT(span<const Index>(effective_layout.read_chunk_shape()),
              ElementsAre(8, 16));
  EXPECT_EQ(effective_layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.grid_origin(), ElementsAre(0, 0));
  EXPECT_EQ(effective_layout.grid_origin().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.inner_order(), ElementsAre(1, 0));
  EXPECT_TRUE(effective_layout.inner_order().hard_constraint);
}

TEST(GetEffectiveChunkLayoutTest, SchemaSoftInnerOrder) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8);
  DimensionIndex rank = 2;
  std::vector<std::string> labels = {"y", "x"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout initial_layout,
      GetInitialChunkLayout(ifd, rank, labels, PlanarConfigType::kChunky, 1,
                            "c"));
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(ChunkLayout::InnerOrder({1, 0}, /*hard=*/false)));
  DimensionSet expected_hard_constraints = DimensionSet::UpTo(rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout effective_layout,
      GetEffectiveChunkLayout(initial_layout, schema));

  EXPECT_THAT(span<const Index>(effective_layout.read_chunk_shape()),
              ElementsAre(8, 16));
  EXPECT_EQ(effective_layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.grid_origin(), ElementsAre(0, 0));
  EXPECT_EQ(effective_layout.grid_origin().hard_constraint,
            expected_hard_constraints);
  EXPECT_THAT(effective_layout.inner_order(), ElementsAre(1, 0));
  EXPECT_FALSE(effective_layout.inner_order().hard_constraint);  // Still soft
}

TEST(GetEffectiveChunkLayoutTest, SchemaSoftChunkShape) {
  ImageDirectory ifd =
      MakeImageDirectory(/*width=*/60, /*height=*/40,
                         /*chunk_width=*/16, /*chunk_height=*/8);
  DimensionIndex rank = 2;
  std::vector<std::string> labels = {"y", "x"};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout initial_layout,
      GetInitialChunkLayout(ifd, rank, labels, PlanarConfigType::kChunky, 1,
                            "c"));
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(ChunkLayout::ReadChunkShape({10, 20}, /*hard=*/false)));
  DimensionSet expected_hard_constraints = DimensionSet::UpTo(rank);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ChunkLayout effective_layout,
      GetEffectiveChunkLayout(initial_layout, schema));

  EXPECT_THAT(span<const Index>(effective_layout.read_chunk_shape()),
              ElementsAre(8, 16));  // Still TIFF shape
  EXPECT_EQ(effective_layout.read_chunk_shape().hard_constraint,
            expected_hard_constraints);  // Still hard
}

// --- GetEffective... tests ---
TEST(GetEffectiveDomainTest, InitialOnly) {
  DimensionIndex rank = 3;
  std::vector<Index> shape = {10, 20, 30};
  std::vector<std::string> labels = {"z", "y", "x"};
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, GetEffectiveDomain(rank, shape, labels, schema));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_domain,
      IndexDomainBuilder(3).shape(shape).labels(labels).Finalize());

  EXPECT_EQ(result.first, expected_domain);
  EXPECT_EQ(result.second, labels);
}

TEST(GetEffectiveDomainTest, SchemaRankOnly) {
  DimensionIndex rank = 3;
  std::vector<Index> shape = {10, 20, 30};
  std::vector<std::string> labels = {"z", "y", "x"};
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(RankConstraint{3}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, GetEffectiveDomain(rank, shape, labels, schema));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_domain,
      IndexDomainBuilder(3).shape(shape).labels(labels).Finalize());

  EXPECT_EQ(result.first, expected_domain);
  EXPECT_EQ(result.second, labels);
}

TEST(GetEffectiveDomainTest, SchemaDomainOverridesLabels) {
  DimensionIndex rank = 3;
  std::vector<Index> shape = {10, 20, 30};
  std::vector<std::string> initial_labels = {"z", "y", "x"};
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto schema_domain,
      IndexDomainBuilder(3).shape(shape).labels({"Z", "Y", "X"}).Finalize());
  TENSORSTORE_ASSERT_OK(schema.Set(schema_domain));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, GetEffectiveDomain(rank, shape, initial_labels, schema));

  EXPECT_EQ(result.first, schema_domain);                  // Domain from schema
  EXPECT_THAT(result.second, ElementsAre("Z", "Y", "X"));  // Labels from schema
}

TEST(GetEffectiveDomainTest, SchemaDomainIncompatibleShape) {
  DimensionIndex rank = 3;
  std::vector<Index> initial_shape = {10, 20, 30};
  std::vector<std::string> initial_labels = {"z", "y", "x"};
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema_domain,
                                   IndexDomainBuilder(3)
                                       .shape({10, 20, 31})
                                       .labels(initial_labels)
                                       .Finalize());
  TENSORSTORE_ASSERT_OK(schema.Set(schema_domain));

  EXPECT_THAT(GetEffectiveDomain(rank, initial_shape, initial_labels, schema),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Mismatch in dimension 2:.*"));
}

TEST(GetEffectiveDomainTest, SchemaRankIncompatible) {
  DimensionIndex rank = 3;
  std::vector<Index> initial_shape = {10, 20, 30};
  std::vector<std::string> initial_labels = {"z", "y", "x"};
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(RankConstraint{2}));  // Rank mismatch

  EXPECT_THAT(GetEffectiveDomain(rank, initial_shape, initial_labels, schema),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*rank constraint 2 is incompatible.*rank 3.*"));
}

TEST(GetEffectiveDimensionUnitsTest, InitialOnly) {
  DimensionIndex rank = 3;
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto units,
                                   GetEffectiveDimensionUnits(rank, schema));
  ASSERT_EQ(units.size(), 3);
  EXPECT_THAT(units, ElementsAre(std::nullopt, std::nullopt, std::nullopt));
}

TEST(GetEffectiveDimensionUnitsTest, SchemaOnly) {
  DimensionIndex rank = 2;
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(Schema::DimensionUnits({"nm", "um"})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto units,
                                   GetEffectiveDimensionUnits(rank, schema));
  ASSERT_EQ(units.size(), 2);
  EXPECT_THAT(units[0], Optional(tensorstore::Unit("nm")));
  EXPECT_THAT(units[1], Optional(tensorstore::Unit("um")));
}

TEST(GetEffectiveDimensionUnitsTest, SchemaRankMismatch) {
  DimensionIndex rank = 3;  // TIFF implies rank 3
  Schema schema;
  TENSORSTORE_ASSERT_OK(
      schema.Set(Schema::DimensionUnits({"nm", "um"})));  // Implies rank 2

  EXPECT_THAT(GetEffectiveDimensionUnits(rank, schema),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Schema dimension_units rank.*"));
}

TEST(GetEffectiveCompressorTest, InitialOnlyRaw) {
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto compressor,
      GetEffectiveCompressor(CompressionType::kNone, schema.codec()));
  EXPECT_EQ(compressor, nullptr);
}

TEST(GetEffectiveCompressorTest, InitialOnlyDeflate) {
  Schema schema;
  EXPECT_THAT(GetEffectiveCompressor(CompressionType::kDeflate, schema.codec()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*deflate.*not registered.*"));
}

TEST(GetEffectiveCompressorTest, SchemaMatchesDeflate) {
  Schema schema;
  TENSORSTORE_ASSERT_OK(schema.Set(
      CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "deflate"}})
          .value()));

  EXPECT_THAT(GetEffectiveCompressor(CompressionType::kDeflate, schema.codec()),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*deflate.*not registered.*"));
}

TEST(GetEffectiveDataTypeTest, ManyChecks) {
  TiffMetadataConstraints constraints;
  Schema schema;
  EXPECT_FALSE(GetEffectiveDataType(constraints, schema).value().valid());
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<uint16_t>));
  EXPECT_THAT(GetEffectiveDataType(constraints, schema),
              Optional(dtype_v<uint16_t>));
  schema = Schema();
  constraints.dtype = dtype_v<float>;
  EXPECT_THAT(GetEffectiveDataType(constraints, schema),
              Optional(dtype_v<float>));
  TENSORSTORE_ASSERT_OK(schema.Set(dtype_v<float>));
  EXPECT_THAT(GetEffectiveDataType(constraints, schema),
              Optional(dtype_v<float>));
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
                    {16, 16, 3}, {0, 1, 2});
  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x", "c"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(0));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(1));
  EXPECT_THAT(metadata->dimension_mapping.ts_sample_dim, Optional(2));
  EXPECT_TRUE(metadata->dimension_mapping.ts_stacked_dims.empty());
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("y", "x", "c"));
}

TEST(ResolveMetadataTest, SelectIfd) {
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 1, 8),
                       MakeImageDirectory(50, 40, 8, 8, true, 3, 16)});
  TiffSpecOptions options;
  options.ifd_index = 1;
  Schema schema;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  CheckBaseMetadata(*metadata, 1, 1, 3, {40, 50, 3}, dtype_v<uint16_t>, 3,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {8, 8, 3}, {0, 1, 2});

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x", "c"));
}

TEST(ResolveMetadataTest, InvalidIfdIndex) {
  auto parse_result = MakeParseResult({MakeImageDirectory()});
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
  // Image is 80x100, spp=3 -> initial order/labels: y, x, c
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 3)});
  TiffSpecOptions options;
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto desired_domain,
                                   IndexDomainBuilder(3)
                                       .shape({80, 100, 3})
                                       .labels({"height", "width", "channel"})
                                       .Finalize());

  TENSORSTORE_ASSERT_OK(schema.Set(desired_domain));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  // Check that ResolveMetadata respected the schema's domain labels
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
  // The default soft inner order is still {0, 1, 2} relative to the final
  // axes
  EXPECT_THAT(metadata->chunk_layout.inner_order(), ElementsAre(0, 1, 2));
}

TEST(ResolveMetadataTest, SchemaUseSampleDimensionLabel) {
  auto parse_result =
      MakeParseResult({MakeImageDirectory(100, 80, 16, 16, true, 3)});
  TiffSpecOptions options;
  options.sample_dimension_label = "comp";
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto desired_domain,
                                   IndexDomainBuilder(3)
                                       .shape({80, 100, 3})
                                       .labels({"y", "x", "comp"})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK(schema.Set(desired_domain));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("y", "x", "comp"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(0));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(1));
  EXPECT_THAT(metadata->dimension_mapping.ts_sample_dim, Optional(2));
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("y", "x", "comp"));
}

TEST(ResolveMetadataTest, StackZ_Spp1) {
  std::vector<ImageDirectory> ifds;
  for (int i = 0; i < 5; ++i)
    ifds.push_back(MakeImageDirectory(32, 64, 8, 16, true, 1));
  auto parse_result = MakeParseResult(ifds);
  TiffSpecOptions options;
  options.ifd_stacking.emplace();
  options.ifd_stacking->dimensions = {"z"};
  options.ifd_stacking->ifd_count = 5;
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  // Default order: Z, Y, X
  CheckBaseMetadata(*metadata, 0, 5, 3, {5, 64, 32}, dtype_v<uint8_t>, 1,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {1, 16, 8}, {0, 1, 2});

  EXPECT_THAT(metadata->dimension_labels, ElementsAre("z", "y", "x"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(1));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(2));
  EXPECT_FALSE(metadata->dimension_mapping.ts_sample_dim.has_value());
  EXPECT_THAT(metadata->dimension_mapping.ts_stacked_dims,
              ElementsAre(testing::Pair("z", 0)));
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("z", "y", "x"));
}

TEST(ResolveMetadataTest, StackTC_Spp3_Chunky) {
  std::vector<ImageDirectory> ifds;
  // 2 time points, 3 channels = 6 IFDs
  for (int i = 0; i < 6; ++i)
    ifds.push_back(MakeImageDirectory(32, 64, 8, 16, true, 3));
  auto parse_result = MakeParseResult(ifds);
  TiffSpecOptions options;
  options.ifd_stacking.emplace();
  options.ifd_stacking->dimensions = {"t", "channel"};
  options.ifd_stacking->dimension_sizes = {2, 3};  // t=2, channel=3 -> 6 IFDs
  options.sample_dimension_label = "rgb";          // Label the SPP dim
  Schema schema;

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, ResolveMetadata(parse_result, options, schema));

  // Default order: T, Channel, Y, X, RGB
  CheckBaseMetadata(*metadata, 0, 6, 5, {2, 3, 64, 32, 3}, dtype_v<uint8_t>, 3,
                    CompressionType::kNone, PlanarConfigType::kChunky,
                    {1, 1, 16, 8, 3}, {0, 1, 2, 3, 4});

  EXPECT_THAT(metadata->dimension_labels,
              ElementsAre("t", "channel", "y", "x", "rgb"));
  EXPECT_THAT(metadata->dimension_mapping.ts_y_dim, Optional(2));
  EXPECT_THAT(metadata->dimension_mapping.ts_x_dim, Optional(3));
  EXPECT_THAT(metadata->dimension_mapping.ts_sample_dim, Optional(4));
  EXPECT_THAT(metadata->dimension_mapping.ts_stacked_dims,
              ::testing::UnorderedElementsAre(testing::Pair("t", 0),
                                              testing::Pair("channel", 1)));
  EXPECT_THAT(metadata->dimension_mapping.labels_by_ts_dim,
              ElementsAre("t", "channel", "y", "x", "rgb"));
}

TEST(ResolveMetadataTest, StackNonUniformIFDs) {
  std::vector<ImageDirectory> ifds;
  ifds.push_back(MakeImageDirectory(32, 64, 8, 16, true, 1));
  ifds.push_back(MakeImageDirectory(32, 64, 8, 16, true, 1));
  ifds.push_back(
      MakeImageDirectory(32, 65, 8, 16, true, 1));  // Different height
  auto parse_result = MakeParseResult(ifds);
  TiffSpecOptions options;
  options.ifd_stacking.emplace();
  options.ifd_stacking->dimensions = {"z"};
  options.ifd_stacking->ifd_count = 3;
  Schema schema;

  EXPECT_THAT(
      ResolveMetadata(parse_result, options, schema),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*IFD 2 dimensions \\(32 x 65\\) do not match IFD 0.*"));
}

// --- Tests for ValidateResolvedMetadata ---
TEST(ValidateResolvedMetadataTest, CompatibleConstraints) {
  auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16, 16)});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ResolveMetadata(parse_result, {}, {}));
  TiffMetadataConstraints constraints;

  TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
  constraints.rank = 2;
  TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
  constraints.rank = dynamic_rank;
  constraints.dtype = dtype_v<uint8_t>;
  TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
  constraints.dtype = std::nullopt;
  constraints.shape = {{80, 100}};
  TENSORSTORE_EXPECT_OK(ValidateResolvedMetadata(*metadata, constraints));
}

TEST(ValidateResolvedMetadataTest, IncompatibleRank) {
  auto parse_result = MakeParseResult({MakeImageDirectory(100, 80, 16, 16)});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   ResolveMetadata(parse_result, {}, {}));
  TiffMetadataConstraints constraints;
  constraints.rank = 3;
  EXPECT_THAT(ValidateResolvedMetadata(*metadata, constraints),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Resolved TIFF rank \\(2\\) does not match user "
                            "constraint rank \\(3\\).*"));
}

// Helper function to encode an array to a Cord for testing DecodeChunk
Result<absl::Cord> EncodeArrayToCord(SharedArrayView<const void> array,
                                     tensorstore::endian source_endian,
                                     ContiguousLayoutOrder order) {
  absl::Cord cord;
  riegeli::CordWriter<> writer(&cord);
  if (!tensorstore::internal::EncodeArrayEndian(array, source_endian, order,
                                                writer)) {
    return writer.status();
  }
  if (!writer.Close()) {
    return writer.status();
  }
  return cord;
}

// Test fixture for DecodeChunk tests
class DecodeChunkTest : public ::testing::Test {
 protected:
  TiffMetadata CreateMetadata(
      DataType dtype, span<const Index> shape,
      span<const Index> grid_chunk_shape,
      ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
      Endian endian = Endian::kLittle,
      CompressionType compression = CompressionType::kNone,
      uint16_t samples_per_pixel = 1,
      PlanarConfigType planar_config = PlanarConfigType::kChunky) {
    TiffMetadata metadata;
    metadata.dtype = dtype;
    metadata.rank = shape.size();
    metadata.shape.assign(shape.begin(), shape.end());
    metadata.endian = endian;
    metadata.compression_type = compression;
    metadata.samples_per_pixel = samples_per_pixel;
    metadata.planar_config = planar_config;
    metadata.compressor = Compressor{nullptr};

    TENSORSTORE_CHECK_OK(
        metadata.chunk_layout.Set(RankConstraint{metadata.rank}));
    TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(
        ChunkLayout::ChunkShape(grid_chunk_shape, /*hard=*/true)));
    TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(ChunkLayout::GridOrigin(
        GetConstantVector<Index, 0>(metadata.rank), /*hard=*/true)));
    std::vector<DimensionIndex> inner_order(metadata.rank);
    tensorstore::SetPermutation(layout_order, span(inner_order));
    TENSORSTORE_CHECK_OK(metadata.chunk_layout.Set(
        ChunkLayout::InnerOrder(inner_order, /*hard=*/true)));
    TENSORSTORE_CHECK_OK(metadata.chunk_layout.Finalize());

    metadata.layout_order = layout_order;

    if (!grid_chunk_shape.empty()) {
      metadata.ifd0_chunk_height =
          (metadata.rank > 0) ? grid_chunk_shape[metadata.rank - 2] : 0;
      // Assuming X is last
      metadata.ifd0_chunk_width =
          (metadata.rank > 0) ? grid_chunk_shape.back() : 0;
      if (planar_config == PlanarConfigType::kPlanar && metadata.rank > 0) {
        metadata.ifd0_chunk_height =
            (metadata.rank > 1) ? grid_chunk_shape[metadata.rank - 2] : 0;  // Y
        metadata.ifd0_chunk_width =
            (metadata.rank > 0) ? grid_chunk_shape.back() : 0;  // X
      }
    }

    return metadata;
  }
};

TEST_F(DecodeChunkTest, UncompressedUint8CorderLittleEndianChunkySpp1) {
  const Index shape[] = {2, 3};
  const Index grid_chunk_shape[] = {2, 3};  // Grid shape matches image shape
  auto metadata = CreateMetadata(
      dtype_v<uint8_t>, shape, grid_chunk_shape, ContiguousLayoutOrder::c,
      Endian::kLittle, CompressionType::kNone, 1, PlanarConfigType::kChunky);
  auto expected_array = MakeArray<uint8_t>({{1, 2, 3}, {4, 5, 6}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::little,
                                         ContiguousLayoutOrder::c));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  SharedArray<const uint8_t> decoded_array(
      std::static_pointer_cast<const uint8_t>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_EQ(decoded_array, expected_array);
}

TEST_F(DecodeChunkTest, UncompressedUint8CorderLittleEndianChunkySpp3) {
  const Index shape[] = {2, 3, 3};             // Y, X, C
  const Index grid_chunk_shape[] = {2, 3, 3};  // Grid shape is Y, X
  const uint16_t spp = 3;
  auto metadata = CreateMetadata(
      dtype_v<uint8_t>, shape, grid_chunk_shape, ContiguousLayoutOrder::c,
      Endian::kLittle, CompressionType::kNone, spp, PlanarConfigType::kChunky);

  auto expected_array = MakeArray<uint8_t>(
      {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
       {{11, 12, 13}, {14, 15, 16}, {17, 18, 19}}});  // Y=2, X=3, C=3
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::little,
                                         ContiguousLayoutOrder::c));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  SharedArray<const uint8_t> decoded_array(
      std::static_pointer_cast<const uint8_t>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_THAT(decoded_array.shape(), ElementsAre(2, 3, 3));
  EXPECT_EQ(decoded_array, expected_array);
}

TEST_F(DecodeChunkTest, UncompressedUint16FortranOrderBigEndian) {
  const Index shape[] = {2, 3};
  const Index grid_chunk_shape[] = {2, 3};
  auto metadata = CreateMetadata(dtype_v<uint16_t>, shape, grid_chunk_shape,
                                 ContiguousLayoutOrder::fortran, Endian::kBig);
  auto expected_array = tensorstore::MakeCopy(
      MakeArray<uint16_t>({{100, 200, 300}, {400, 500, 600}}),
      ContiguousLayoutOrder::fortran);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::big,
                                         ContiguousLayoutOrder::fortran));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  SharedArray<const uint16_t> decoded_array(
      std::static_pointer_cast<const uint16_t>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_EQ(decoded_array, expected_array);
}

TEST_F(DecodeChunkTest, UncompressedFloat32CorderBigEndianToNative) {
  const Index shape[] = {2, 2};
  // Native endian might be little, source is big
  auto metadata = CreateMetadata(dtype_v<float>, shape, shape,
                                 ContiguousLayoutOrder::c, Endian::kBig);
  auto expected_array = MakeArray<float>({{1.0f, 2.5f}, {-3.0f, 4.75f}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord,
      EncodeArrayToCord(expected_array, endian::big, ContiguousLayoutOrder::c));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  SharedArray<const float> decoded_array(
      std::static_pointer_cast<const float>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_EQ(decoded_array, expected_array);
}

TEST_F(DecodeChunkTest, UncompressedRank3) {
  const Index shape[] = {2, 3, 2};  // Y, X, C
  auto metadata = CreateMetadata(dtype_v<int16_t>, shape, shape,
                                 ContiguousLayoutOrder::c, Endian::kLittle);
  auto expected_array = MakeArray<int16_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::little,
                                         ContiguousLayoutOrder::c));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  SharedArray<const int16_t> decoded_array(
      std::static_pointer_cast<const int16_t>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_EQ(decoded_array, expected_array);
}

TEST_F(DecodeChunkTest, ErrorInputTooSmall) {
  const Index shape[] = {2, 3};
  auto metadata = CreateMetadata(dtype_v<uint16_t>, shape, shape,
                                 ContiguousLayoutOrder::c, Endian::kLittle);
  auto expected_array = MakeArray<uint16_t>({{1, 2, 3}, {4, 5, 6}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::little,
                                         ContiguousLayoutOrder::c));

  // Truncate the cord
  absl::Cord truncated_cord = input_cord.Subcord(0, input_cord.size() - 1);

  EXPECT_THAT(
      DecodeChunk(metadata, truncated_cord),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*Not enough data.*"));
}

TEST_F(DecodeChunkTest, ErrorExcessData) {
  const Index shape[] = {2, 3};
  auto metadata = CreateMetadata(dtype_v<uint8_t>, shape, shape,
                                 ContiguousLayoutOrder::c, Endian::kLittle);
  auto expected_array = MakeArray<uint8_t>({{1, 2, 3}, {4, 5, 6}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto input_cord, EncodeArrayToCord(expected_array, endian::little,
                                         ContiguousLayoutOrder::c));

  // Add extra data
  input_cord.Append("extra");

  EXPECT_THAT(DecodeChunk(metadata, input_cord),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*End of data expected.*"));
}

// --- Placeholder Tests for Compression ---
// These require compressor implementations to be registered and
// potentially pre-compressed "golden" data.
TEST_F(DecodeChunkTest, DISABLED_CompressedDeflate) {
  // 1. Register Deflate compressor (implementation needed separately)
  //    RegisterTiffCompressor<DeflateCompressor>("deflate", ...);

  // 2. Create metadata with deflate compression
  const Index shape[] = {4, 5};
  auto metadata =
      CreateMetadata(dtype_v<uint16_t>, shape, shape, ContiguousLayoutOrder::c,
                     Endian::kLittle, CompressionType::kDeflate);
  // Get compressor instance via ResolveMetadata or manually for test
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      metadata.compressor,
      Compressor::FromJson({{"type", "deflate"}}));  // Assumes registration

  // 3. Create expected *decoded* array
  auto expected_array = AllocateArray<uint16_t>(shape, ContiguousLayoutOrder::c,
                                                tensorstore::value_init);
  // Fill with some data...
  for (Index i = 0; i < 4; ++i)
    for (Index j = 0; j < 5; ++j) expected_array(i, j) = i * 10 + j;

  // 4. Create *compressed* input cord (requires deflate implementation or
  // golden data) Example using golden data (replace hex string with actual
  // compressed bytes) std::string compressed_hex = "789c...";
  // TENSORSTORE_ASSERT_OK_AND_ASSIGN(absl::Cord input_cord,
  // HexToCord(compressed_hex));
  absl::Cord input_cord;  // Placeholder - needs real compressed data
  GTEST_SKIP()
      << "Skipping compressed test until compressor impl/data is available.";

  // 5. Call DecodeChunk and verify
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_array_void,
                                   DecodeChunk(metadata, input_cord));
  // Cast the void result to the expected type, preserving layout
  SharedArray<const uint16_t> decoded_array(
      std::static_pointer_cast<const uint16_t>(decoded_array_void.pointer()),
      expected_array.layout());

  EXPECT_EQ(decoded_array, expected_array);
}

}  // namespace