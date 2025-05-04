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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/tiff/tiff_test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {
namespace kvstore = tensorstore::kvstore;

using ::tensorstore::CodecSpec;
using ::tensorstore::Context;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dtype_v;
using ::tensorstore::GetMemoryKeyValueStore;
using ::tensorstore::Index;
using ::tensorstore::kImplicit;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Schema;
using ::tensorstore::Spec;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal_tiff_kvstore::testing::PutLE16;
using ::tensorstore::internal_tiff_kvstore::testing::PutLE32;
using ::tensorstore::internal_tiff_kvstore::testing::TiffBuilder;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Optional;

class TiffDriverTest : public ::testing::Test {
 protected:
  Context context_ = Context::Default();

  // Helper to write float bytes in Little Endian
  static void PutLEFloat32(std::string& dst, float f) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    uint32_t bits;
    // issues
    std::memcpy(&bits, &f, sizeof(float));
    PutLE32(dst, bits);
  }

  // Helper to write TIFF data to memory kvstore
  void WriteTiffData(std::string_view key, const std::string& tiff_data) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        tensorstore::KvStore store,
        kvstore::Open({{"driver", "memory"}}, context_).result());
    TENSORSTORE_ASSERT_OK(kvstore::Write(store, key, absl::Cord(tiff_data)));
  }

  std::string MakeMinimalTiff() {
    // 10x20 uint8, 1 channel, chunky, 10x10 tiles
    TiffBuilder builder;
    builder.StartIfd(10)
        .AddEntry(256, 3, 1, 10)   // ImageWidth = 10
        .AddEntry(257, 3, 1, 20)   // ImageLength = 20
        .AddEntry(277, 3, 1, 1)    // SamplesPerPixel = 1
        .AddEntry(258, 3, 1, 8)    // BitsPerSample = 8
        .AddEntry(259, 3, 1, 1)    // Compression = None
        .AddEntry(262, 3, 1, 1)    // PhotometricInterpretation = MinIsBlack
        .AddEntry(322, 3, 1, 10)   // TileWidth = 10
        .AddEntry(323, 3, 1, 10);  // TileLength = 10
    // Fake tile data offsets/counts
    size_t data_start = builder.CurrentOffset() + 12 * 9 + 4 + 4 * 4;
    builder.AddEntry(324, 4, 2, builder.CurrentOffset() + 12 * 9 + 4);
    builder.AddEntry(325, 4, 2, builder.CurrentOffset() + 12 * 9 + 4 + 4 * 2);
    builder.EndIfd(0);
    builder.AddUint32Array(
        {(uint32_t)data_start, (uint32_t)(data_start + 100)});
    builder.AddUint32Array({100, 100});
    builder.data_.append(100, '\1');
    builder.data_.append(100, '\2');
    return builder.Build();
  }

  std::string MakeReadTestTiff() {
    // 4x6 uint16, 1 channel, chunky, 2x3 tiles
    std::vector<uint16_t> tile0_data = {1, 2, 3, 7, 8, 9};
    std::vector<uint16_t> tile1_data = {4, 5, 6, 10, 11, 12};
    std::vector<uint16_t> tile2_data = {13, 14, 15, 19, 20, 21};
    std::vector<uint16_t> tile3_data = {16, 17, 18, 22, 23, 24};
    size_t tile_size_bytes = 6 * sizeof(uint16_t);

    TiffBuilder builder;
    builder.StartIfd(10)
        .AddEntry(256, 3, 1, 6)   // Width = 6
        .AddEntry(257, 3, 1, 4)   // Height = 4
        .AddEntry(277, 3, 1, 1)   // SamplesPerPixel = 1
        .AddEntry(258, 3, 1, 16)  // BitsPerSample = 16
        .AddEntry(259, 3, 1, 1)   // Compression = None
        .AddEntry(262, 3, 1, 1)   // Photometric = MinIsBlack
        .AddEntry(322, 3, 1, 3)   // TileWidth = 3
        .AddEntry(323, 3, 1, 2);  // TileLength = 2

    size_t header_size = 8;
    size_t ifd_block_size = 2 + (10 * 12) + 4;
    size_t end_of_ifd_offset = header_size + ifd_block_size;

    size_t tile_offsets_array_start_offset = end_of_ifd_offset;
    size_t tile_offsets_array_size = 4 * sizeof(uint32_t);
    size_t tile_bytecounts_array_start_offset =
        tile_offsets_array_start_offset + tile_offsets_array_size;
    size_t tile_bytecounts_array_size = 4 * sizeof(uint32_t);
    size_t tile_data_start_offset =
        tile_bytecounts_array_start_offset + tile_bytecounts_array_size;

    std::vector<uint32_t> tile_offsets = {
        (uint32_t)(tile_data_start_offset + 0 * tile_size_bytes),
        (uint32_t)(tile_data_start_offset + 1 * tile_size_bytes),
        (uint32_t)(tile_data_start_offset + 2 * tile_size_bytes),
        (uint32_t)(tile_data_start_offset + 3 * tile_size_bytes)};
    std::vector<uint32_t> tile_bytecounts(4, tile_size_bytes);

    builder.AddEntry(324, 4, tile_offsets.size(),
                     tile_offsets_array_start_offset);
    builder.AddEntry(325, 4, tile_bytecounts.size(),
                     tile_bytecounts_array_start_offset);

    builder.EndIfd(0)
        .AddUint32Array(tile_offsets)
        .AddUint32Array(tile_bytecounts);

    auto append_tile = [&](const std::vector<uint16_t>& data) {
      for (uint16_t val : data) {
        PutLE16(builder.data_, val);
      }
    };
    append_tile(tile0_data);
    append_tile(tile1_data);
    append_tile(tile2_data);
    append_tile(tile3_data);

    return builder.Build();
  }

  // Generates a 6x8 uint8 image with 3 strips (RowsPerStrip = 2)
  std::string MakeStrippedTiff() {
    const uint32_t image_width = 8;
    const uint32_t image_height = 6;
    const uint32_t rows_per_strip = 2;
    const uint32_t num_strips =
        (image_height + rows_per_strip - 1) / rows_per_strip;
    const uint32_t bytes_per_strip =
        rows_per_strip * image_width * sizeof(uint8_t);

    const uint16_t num_ifd_entries = 10;

    TiffBuilder builder;
    builder.StartIfd(num_ifd_entries)
        .AddEntry(256, 3, 1, image_width)   // ImageWidth
        .AddEntry(257, 3, 1, image_height)  // ImageLength
        .AddEntry(277, 3, 1, 1)             // SamplesPerPixel = 1
        .AddEntry(258, 3, 1, 8)             // BitsPerSample = 8
        .AddEntry(339, 3, 1, 1)             // SampleFormat = uint
        .AddEntry(259, 3, 1, 1)             // Compression = None
        .AddEntry(262, 3, 1, 1)  // PhotometricInterpretation = MinIsBlack
        .AddEntry(278, 3, 1, rows_per_strip);  // RowsPerStrip

    size_t header_size = 8;
    size_t ifd_block_size = 2 + (num_ifd_entries * 12) + 4;
    size_t end_of_ifd_offset = header_size + ifd_block_size;

    size_t strip_offsets_array_start_offset = end_of_ifd_offset;
    size_t strip_offsets_array_size = num_strips * sizeof(uint32_t);
    size_t strip_bytecounts_array_start_offset =
        strip_offsets_array_start_offset + strip_offsets_array_size;
    size_t strip_bytecounts_array_size = num_strips * sizeof(uint32_t);
    size_t strip_data_start_offset =
        strip_bytecounts_array_start_offset + strip_bytecounts_array_size;

    std::vector<uint32_t> strip_offsets;
    std::vector<uint32_t> strip_bytecounts;
    for (uint32_t i = 0; i < num_strips; ++i) {
      strip_offsets.push_back(strip_data_start_offset + i * bytes_per_strip);
      strip_bytecounts.push_back(bytes_per_strip);
    }

    builder.AddEntry(273, 4, strip_offsets.size(),
                     strip_offsets_array_start_offset);
    builder.AddEntry(279, 4, strip_bytecounts.size(),
                     strip_bytecounts_array_start_offset);

    builder.EndIfd(0)
        .AddUint32Array(strip_offsets)
        .AddUint32Array(strip_bytecounts);

    for (uint32_t s = 0; s < num_strips; ++s) {
      for (uint32_t i = 0; i < bytes_per_strip; ++i) {
        builder.data_.push_back(static_cast<char>(s * 10 + i));
      }
    }

    return builder.Build();
  }

  // Generates a 2x3 float32 image with 1x1 tiles
  std::string MakeFloatTiff() {
    const uint32_t image_width = 3;
    const uint32_t image_height = 2;
    const uint32_t tile_width = 1;
    const uint32_t tile_height = 1;
    const uint32_t num_tiles =
        (image_height / tile_height) * (image_width / tile_width);
    const uint32_t bytes_per_tile = tile_height * tile_width * sizeof(float);

    const uint16_t num_ifd_entries = 11;

    TiffBuilder builder;
    builder.StartIfd(num_ifd_entries)
        .AddEntry(256, 3, 1, image_width)   // ImageWidth
        .AddEntry(257, 3, 1, image_height)  // ImageLength
        .AddEntry(277, 3, 1, 1)             // SamplesPerPixel = 1
        .AddEntry(258, 3, 1, 32)            // BitsPerSample = 32
        .AddEntry(339, 3, 1, 3)             // SampleFormat = IEEEFloat (3)
        .AddEntry(259, 3, 1, 1)             // Compression = None
        .AddEntry(262, 3, 1, 1)  // PhotometricInterpretation = MinIsBlack
        .AddEntry(322, 3, 1, tile_width)    // TileWidth
        .AddEntry(323, 3, 1, tile_height);  // TileLength

    size_t header_size = 8;
    size_t ifd_block_size = 2 + (num_ifd_entries * 12) + 4;
    size_t end_of_ifd_offset = header_size + ifd_block_size;

    size_t tile_offsets_array_start_offset = end_of_ifd_offset;
    size_t tile_offsets_array_size = num_tiles * sizeof(uint32_t);
    size_t tile_bytecounts_array_start_offset =
        tile_offsets_array_start_offset + tile_offsets_array_size;
    size_t tile_bytecounts_array_size = num_tiles * sizeof(uint32_t);
    size_t tile_data_start_offset =
        tile_bytecounts_array_start_offset + tile_bytecounts_array_size;

    std::vector<uint32_t> tile_offsets;
    std::vector<uint32_t> tile_bytecounts;
    for (uint32_t i = 0; i < num_tiles; ++i) {
      tile_offsets.push_back(tile_data_start_offset + i * bytes_per_tile);
      tile_bytecounts.push_back(bytes_per_tile);
    }

    builder.AddEntry(324, 4, tile_offsets.size(),
                     tile_offsets_array_start_offset);
    builder.AddEntry(325, 4, tile_bytecounts.size(),
                     tile_bytecounts_array_start_offset);

    builder.EndIfd(0)
        .AddUint32Array(tile_offsets)
        .AddUint32Array(tile_bytecounts);

    const std::vector<float> values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
    for (float val : values) {
      PutLEFloat32(builder.data_, val);
    }

    return builder.Build();
  }

  // Generates a 2x3 uint8 RGB image with 1x1 tiles (Chunky config)
  std::string MakeMultiChannelTiff() {
    const uint32_t image_width = 3;
    const uint32_t image_height = 2;
    const uint32_t samples_per_pixel = 3;  // RGB
    const uint32_t tile_width = 1;
    const uint32_t tile_height = 1;
    const uint32_t num_tiles =
        (image_height / tile_height) * (image_width / tile_width);
    const uint32_t bytes_per_tile =
        tile_height * tile_width * samples_per_pixel * sizeof(uint8_t);

    const uint16_t num_ifd_entries = 12;

    std::vector<uint16_t> bits_per_sample_data = {8, 8, 8};
    std::vector<uint16_t> sample_format_data = {1, 1, 1};

    TiffBuilder builder;
    builder.StartIfd(num_ifd_entries)
        .AddEntry(256, 3, 1, image_width)        // ImageWidth
        .AddEntry(257, 3, 1, image_height)       // ImageLength
        .AddEntry(277, 3, 1, samples_per_pixel)  // SamplesPerPixel
        .AddEntry(284, 3, 1, 1)           // PlanarConfiguration = Chunky (1)
        .AddEntry(259, 3, 1, 1)           // Compression = None
        .AddEntry(262, 3, 1, 2)           // PhotometricInterpretation = RGB (2)
        .AddEntry(322, 3, 1, tile_width)  // TileWidth
        .AddEntry(323, 3, 1, tile_height);  // TileLength

    size_t header_size = 8;
    size_t ifd_block_size = 2 + (num_ifd_entries * 12) + 4;
    size_t current_offset = header_size + ifd_block_size;
    size_t bps_array_offset = current_offset;
    size_t bps_array_size = bits_per_sample_data.size() * sizeof(uint16_t);
    current_offset += bps_array_size;

    size_t sf_array_offset = current_offset;
    size_t sf_array_size = sample_format_data.size() * sizeof(uint16_t);
    current_offset += sf_array_size;

    size_t tile_offsets_array_offset = current_offset;
    size_t tile_offsets_array_size = num_tiles * sizeof(uint32_t);
    current_offset += tile_offsets_array_size;

    size_t tile_bytecounts_array_offset = current_offset;
    size_t tile_bytecounts_array_size = num_tiles * sizeof(uint32_t);
    current_offset += tile_bytecounts_array_size;

    size_t tile_data_start_offset = current_offset;

    std::vector<uint32_t> tile_offsets;
    std::vector<uint32_t> tile_bytecounts;
    for (uint32_t i = 0; i < num_tiles; ++i) {
      tile_offsets.push_back(tile_data_start_offset + i * bytes_per_tile);
      tile_bytecounts.push_back(bytes_per_tile);
    }

    builder.AddEntry(258, 3, samples_per_pixel, bps_array_offset);
    builder.AddEntry(339, 3, samples_per_pixel, sf_array_offset);
    builder.AddEntry(324, 4, tile_offsets.size(), tile_offsets_array_offset);
    builder.AddEntry(325, 4, tile_bytecounts.size(),
                     tile_bytecounts_array_offset);

    builder.EndIfd(0);

    builder.AddUint16Array(bits_per_sample_data);
    builder.AddUint16Array(sample_format_data);
    builder.AddUint32Array(tile_offsets);
    builder.AddUint32Array(tile_bytecounts);

    const std::vector<uint8_t> tile_values = {
        1, 2, 3, 2, 3, 4, 3, 4, 5, 11, 12, 13, 12, 13, 14, 13, 14, 15,
    };
    for (uint8_t val : tile_values) {
      builder.data_.push_back(static_cast<char>(val));
    }

    return builder.Build();
  }

  // Generates a TIFF with two IFDs:
  // IFD 0: 2x2 uint8 image, filled with 5
  // IFD 1: 3x3 uint16 image, filled with 99
  std::string MakeMultiIFDTiff() {
    TiffBuilder builder;

    const uint32_t ifd0_width = 2;
    const uint32_t ifd0_height = 2;
    const uint32_t ifd0_num_tiles = 4;
    const uint32_t ifd0_bytes_per_tile = 1 * 1 * 1 * sizeof(uint8_t);
    const uint16_t ifd0_num_entries = 11;
    std::vector<uint8_t> ifd0_pixel_data(ifd0_num_tiles * ifd0_bytes_per_tile,
                                         5);

    const uint32_t ifd1_width = 3;
    const uint32_t ifd1_height = 3;
    const uint32_t ifd1_num_tiles = 9;
    const uint32_t ifd1_bytes_per_tile = 1 * 1 * 1 * sizeof(uint16_t);  // 2
    const uint16_t ifd1_num_entries = 11;
    std::vector<uint16_t> ifd1_pixel_data(
        ifd1_num_tiles * (ifd1_bytes_per_tile / sizeof(uint16_t)), 99);

    size_t header_size = 8;
    size_t ifd0_block_size = 2 + ifd0_num_entries * 12 + 4;
    size_t ifd1_block_size = 2 + ifd1_num_entries * 12 + 4;

    size_t ifd0_start_offset = header_size;
    size_t ifd1_start_offset = ifd0_start_offset + ifd0_block_size;
    size_t end_of_ifds_offset = ifd1_start_offset + ifd1_block_size;

    size_t ifd0_offsets_loc = end_of_ifds_offset;
    size_t ifd0_offsets_size = ifd0_num_tiles * sizeof(uint32_t);
    size_t ifd0_counts_loc = ifd0_offsets_loc + ifd0_offsets_size;
    size_t ifd0_counts_size = ifd0_num_tiles * sizeof(uint32_t);
    size_t ifd0_data_loc = ifd0_counts_loc + ifd0_counts_size;
    size_t ifd0_data_size = ifd0_pixel_data.size();
    size_t ifd1_offsets_loc = ifd0_data_loc + ifd0_data_size;
    size_t ifd1_offsets_size = ifd1_num_tiles * sizeof(uint32_t);
    size_t ifd1_counts_loc = ifd1_offsets_loc + ifd1_offsets_size;
    size_t ifd1_counts_size = ifd1_num_tiles * sizeof(uint32_t);
    size_t ifd1_data_loc = ifd1_counts_loc + ifd1_counts_size;

    std::vector<uint32_t> ifd0_tile_offsets;
    std::vector<uint32_t> ifd0_tile_counts;
    for (uint32_t i = 0; i < ifd0_num_tiles; ++i) {
      ifd0_tile_offsets.push_back(ifd0_data_loc + i * ifd0_bytes_per_tile);
      ifd0_tile_counts.push_back(ifd0_bytes_per_tile);
    }

    std::vector<uint32_t> ifd1_tile_offsets;
    std::vector<uint32_t> ifd1_tile_counts;
    for (uint32_t i = 0; i < ifd1_num_tiles; ++i) {
      ifd1_tile_offsets.push_back(ifd1_data_loc + i * ifd1_bytes_per_tile);
      ifd1_tile_counts.push_back(ifd1_bytes_per_tile);
    }

    builder.StartIfd(ifd0_num_entries)
        .AddEntry(256, 3, 1, ifd0_width)
        .AddEntry(257, 3, 1, ifd0_height)
        .AddEntry(277, 3, 1, 1)
        .AddEntry(258, 3, 1, 8)
        .AddEntry(339, 3, 1, 1)
        .AddEntry(259, 3, 1, 1)
        .AddEntry(262, 3, 1, 1)
        .AddEntry(322, 3, 1, 1)
        .AddEntry(323, 3, 1, 1)
        .AddEntry(324, 4, ifd0_num_tiles, ifd0_offsets_loc)
        .AddEntry(325, 4, ifd0_num_tiles, ifd0_counts_loc);
    builder.EndIfd(ifd1_start_offset);

    builder.PadTo(ifd1_start_offset);
    builder.StartIfd(ifd1_num_entries)
        .AddEntry(256, 3, 1, ifd1_width)
        .AddEntry(257, 3, 1, ifd1_height)
        .AddEntry(277, 3, 1, 1)
        .AddEntry(258, 3, 1, 16)
        .AddEntry(339, 3, 1, 1)
        .AddEntry(259, 3, 1, 1)
        .AddEntry(262, 3, 1, 1)
        .AddEntry(322, 3, 1, 1)
        .AddEntry(323, 3, 1, 1)
        .AddEntry(324, 4, ifd1_num_tiles, ifd1_offsets_loc)
        .AddEntry(325, 4, ifd1_num_tiles, ifd1_counts_loc);
    builder.EndIfd(0);

    builder.PadTo(end_of_ifds_offset);
    builder.AddUint32Array(ifd0_tile_offsets);
    builder.AddUint32Array(ifd0_tile_counts);

    for (uint8_t val : ifd0_pixel_data) {
      builder.data_.push_back(static_cast<char>(val));
    }

    builder.AddUint32Array(ifd1_tile_offsets);
    builder.AddUint32Array(ifd1_tile_counts);

    for (uint16_t val : ifd1_pixel_data) {
      PutLE16(builder.data_, val);
    }

    return builder.Build();
  }
};

// --- Spec Tests ---
TEST_F(TiffDriverTest, SpecFromJsonMinimal) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson({{"driver", "tiff"}, {"kvstore", "memory://test/"}}));
  EXPECT_TRUE(spec.valid());
}

TEST_F(TiffDriverTest, SpecToJsonMinimal) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson({{"driver", "tiff"}, {"kvstore", "memory://test/"}}));
  // Convert back to JSON using default options (excludes defaults)
  EXPECT_THAT(spec.ToJson(),
              Optional(MatchesJson(
                  {{"driver", "tiff"},
                   {"kvstore", {{"driver", "memory"}, {"path", "test/"}}}})));
}

TEST_F(TiffDriverTest, SpecFromJsonWithOptions) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson(
          {{"driver", "tiff"},
           {"kvstore", "memory://test/"},
           {"tiff", {{"ifd", 5}}},
           {"metadata", {{"dtype", "uint16"}, {"shape", {30, 40}}}}}));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, spec.schema());
  EXPECT_EQ(dtype_v<uint16_t>, schema.dtype());
  EXPECT_EQ(2, schema.rank());
}

TEST_F(TiffDriverTest, SpecToJsonWithOptions) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson(
          {{"driver", "tiff"},
           {"kvstore", "memory://test/"},
           {"tiff", {{"ifd", 5}}},
           {"metadata", {{"dtype", "uint16"}, {"shape", {30, 40}}}}}));

  ::nlohmann::json expected_json = {
      {"driver", "tiff"},
      {"kvstore",
       {{"driver", "memory"},
        {"path", "test/"},
        {"atomic", true},
        {"memory_key_value_store", "memory_key_value_store"},
        {"context", ::nlohmann::json({})}}},
      {"dtype", "uint16"},
      {"schema", {{"dtype", "uint16"}, {"rank", 2}}},
      {"transform",
       {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {30, 40}}}},
      {"context", ::nlohmann::json({})},
      {"cache_pool", "cache_pool"},
      {"data_copy_concurrency", "data_copy_concurrency"},
      {"recheck_cached_data", true},
      {"recheck_cached_metadata", "open"},
      {"delete_existing", false},
      {"assume_metadata", false},
      {"assume_cached_metadata", false},
      {"fill_missing_data_reads", true},
      {"store_data_equal_to_fill_value", false},
      {"tiff", {{"ifd", 5}}},
      {"metadata", {{"dtype", "uint16"}, {"shape", {30, 40}}}}};

  EXPECT_THAT(spec.ToJson(tensorstore::IncludeDefaults{true}),
              Optional(MatchesJson(expected_json)));
}

TEST_F(TiffDriverTest, InvalidSpecExtraMember) {
  EXPECT_THAT(
      Spec::FromJson(
          {{"driver", "tiff"}, {"kvstore", "memory://"}, {"extra", "member"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Object includes extra members: \"extra\""));
}

TEST_F(TiffDriverTest, TestSpecSchemaDtype) {
  TestSpecSchema({{"driver", "tiff"},
                  {"kvstore", "memory://"},
                  {"metadata", {{"dtype", "uint16"}}}},
                 // Expected schema now includes the default codec:
                 {{"dtype", "uint16"}, {"codec", {{"driver", "tiff"}}}});
}

TEST_F(TiffDriverTest, TestSpecSchemaRank) {
  // Test that specifying shape infers rank, domain, and default layout/codec
  TestSpecSchema(
      {{"driver", "tiff"},
       {"kvstore", "memory://"},
       {"metadata", {{"shape", {10, 20, 30}}}}},
      {{"rank", 3},
       {"domain",
        {{"inclusive_min", {0, 0, 0}}, {"exclusive_max", {10, 20, 30}}}},
       {"codec", {{"driver", "tiff"}}}});
}

// --- Open Tests ---
TEST_F(TiffDriverTest, InvalidOpenMissingKvstore) {
  // FromJson should succeed structurally, even if kvstore is missing.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   Spec::FromJson({{"driver", "tiff"}}));

  // The Open operation should fail because kvstore is missing/invalid.
  EXPECT_THAT(tensorstore::Open(spec, context_).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*\"kvstore\" must be specified.*"));
}

TEST_F(TiffDriverTest, OpenNonExisting) {
  EXPECT_THAT(tensorstore::Open(
                  {{"driver", "tiff"}, {"kvstore", "memory://nonexistent.tif"}},
                  context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kNotFound, ".*File not found.*"));
}

TEST_F(TiffDriverTest, OpenMinimalTiff) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {
              {"driver", "tiff"},
              {"kvstore", {{"driver", "memory"}, {"path", "minimal.tif"}}},
          },
          context_)
          .result());

  EXPECT_EQ(dtype_v<uint8_t>, store.dtype());
  EXPECT_EQ(2, store.rank());
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(20, 10));
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));

  // Check chunk layout derived from TIFF tags
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(10, 10));
}

TEST_F(TiffDriverTest, OpenWithMatchingMetadataConstraint) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  TENSORSTORE_EXPECT_OK(
      tensorstore::Open(
          {{"driver", "tiff"},
           {"kvstore", "memory://minimal.tif"},
           // Check that constraints match what's in the file
           {"metadata", {{"dtype", "uint8"}, {"shape", {20, 10}}}}},
          context_)
          .result());
}

TEST_F(TiffDriverTest, OpenWithMismatchedDtypeConstraint) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(tensorstore::Open({{"driver", "tiff"},
                                 {"kvstore", "memory://minimal.tif"},
                                 {"metadata", {{"dtype", "uint16"}}}},
                                context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*dtype.*uint16.* conflicts.*uint8.*"));
}

TEST_F(TiffDriverTest, OpenWithMismatchedShapeConstraint) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(tensorstore::Open({{"driver", "tiff"},
                                 {"kvstore", "memory://minimal.tif"},
                                 {"metadata", {{"shape", {20, 11}}}}},
                                context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Resolved TIFF shape .*20, 10.* does not match "
                            "user constraint shape .*20, 11.*"));
}

TEST_F(TiffDriverTest, OpenWithSchemaDtypeMismatch) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(
      tensorstore::Open({{"driver", "tiff"},
                         {"kvstore", "memory://minimal.tif"},
                         {"schema", {{"dtype", "int16"}}}},
                        context_)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*dtype specified in schema.*int16.* conflicts .* dtype .*uint8.*"));
}

TEST_F(TiffDriverTest, OpenInvalidTiffHeader) {
  WriteTiffData("invalid_header.tif", "Not a valid TIFF file");
  EXPECT_THAT(tensorstore::Open({{"driver", "tiff"},
                                 {"kvstore", "memory://invalid_header.tif"}},
                                context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Invalid TIFF byte order mark.*"));
}

TEST_F(TiffDriverTest, OpenInvalidIfdIndex) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(tensorstore::Open({{"driver", "tiff"},
                                 {"kvstore", "memory://minimal.tif"},
                                 {"tiff", {{"ifd", 1}}}},
                                context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kNotFound,
                            ".*Requested IFD index 1 not found.*"));
}

// --- Read Tests ---
TEST_F(TiffDriverTest, ReadFull) {
  WriteTiffData("read_test.tif", MakeReadTestTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "tiff"}, {"kvstore", "memory://read_test.tif"}}, context_)
          .result());

  EXPECT_THAT(
      tensorstore::Read(store).result(),
      Optional(tensorstore::MakeArray<uint16_t>({{1, 2, 3, 4, 5, 6},
                                                 {7, 8, 9, 10, 11, 12},
                                                 {13, 14, 15, 16, 17, 18},
                                                 {19, 20, 21, 22, 23, 24}})));
}

TEST_F(TiffDriverTest, ReadSlice) {
  WriteTiffData("read_test.tif", MakeReadTestTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "tiff"}, {"kvstore", "memory://read_test.tif"}}, context_)
          .result());

  // Read a slice covering parts of tiles 0 and 1
  EXPECT_THAT(
      tensorstore::Read(store | tensorstore::Dims(0, 1).IndexSlice({1, 2}))
          .result(),
      Optional(tensorstore::MakeScalarArray<uint16_t>(9)));

  // Read a slice within a single tile (tile 2)
  EXPECT_THAT(
      tensorstore::Read(store |
                        tensorstore::Dims(0, 1).SizedInterval({2, 1}, {1, 2}))
          .result(),
      Optional(tensorstore::MakeOffsetArray<uint16_t>({2, 1}, {{14, 15}})));
}

// --- Metadata / Property Tests ---
TEST_F(TiffDriverTest, Properties) {
  WriteTiffData("read_test.tif", MakeReadTestTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "tiff"}, {"kvstore", "memory://read_test.tif"}}, context_)
          .result());

  EXPECT_EQ(dtype_v<uint16_t>, store.dtype());
  EXPECT_EQ(2, store.rank());
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(4, 6));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(2, 3));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto codec, store.codec());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_codec,
      CodecSpec::FromJson({{"driver", "tiff"}, {"compression", "raw"}}));
  EXPECT_EQ(expected_codec, codec);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto units, store.dimension_units());
  EXPECT_THAT(units, ::testing::ElementsAre(std::nullopt, std::nullopt));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto fill_value, store.fill_value());
  EXPECT_FALSE(fill_value.valid());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved_store,
                                   ResolveBounds(store).result());
  EXPECT_EQ(store.domain(), resolved_store.domain());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto bound_spec, store.spec());
  ASSERT_TRUE(bound_spec.valid());

  // Check the minimal JSON representation (IncludeDefaults=false)
  ::nlohmann::json expected_minimal_json = {
      {"driver", "tiff"},
      {"kvstore", {{"driver", "memory"}, {"path", "read_test.tif"}}},
      {"dtype", "uint16"},
      {"transform",
       {// Includes the resolved domain
        {"input_labels", {"y", "x"}},
        {"input_inclusive_min", {0, 0}},
        {"input_exclusive_max", {4, 6}}}},
      {"metadata", {{"dtype", "uint16"}, {"shape", {4, 6}}}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto minimal_json, bound_spec.ToJson());
  EXPECT_THAT(minimal_json, MatchesJson(expected_minimal_json));

  // Check the full JSON representation (IncludeDefaults=true)
  ::nlohmann::json expected_full_json = {
      {"driver", "tiff"},
      {"kvstore",
       {{"driver", "memory"},
        {"path", "read_test.tif"},
        {"atomic", true},
        {"context", ::nlohmann::json({})},
        {"memory_key_value_store", "memory_key_value_store"}}},
      {"dtype", "uint16"},
      {"transform",
       {{"input_inclusive_min", {0, 0}},
        {"input_exclusive_max", {4, 6}},
        {"input_labels", {"y", "x"}}}},
      {"metadata", {{"dtype", "uint16"}, {"shape", {4, 6}}}},
      {"tiff", {{"ifd", 0}}},  // Default ifd included
      {"schema", {{"rank", 2}, {"dtype", "uint16"}}},
      {"recheck_cached_data", true},
      {"recheck_cached_metadata", "open"},
      {"delete_existing", false},
      {"assume_metadata", false},
      {"assume_cached_metadata", false},
      {"fill_missing_data_reads", true},
      {"store_data_equal_to_fill_value", false},
      {"cache_pool", "cache_pool"},
      {"context", ::nlohmann::json({})},
      {"data_copy_concurrency", "data_copy_concurrency"}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto full_json, bound_spec.ToJson(tensorstore::IncludeDefaults{true}));
  EXPECT_THAT(full_json, MatchesJson(expected_full_json));

  // Test re-opening from the minimal spec
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2, tensorstore::Open(bound_spec, context_).result());
  EXPECT_EQ(store.dtype(), store2.dtype());
  EXPECT_EQ(store.domain(), store2.domain());
  EXPECT_EQ(store.rank(), store2.rank());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout2, store2.chunk_layout());
  EXPECT_EQ(layout, layout2);
}

TEST_F(TiffDriverTest, ReadStrippedTiff) {
  WriteTiffData("stripped.tif", MakeStrippedTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "tiff"}, {"kvstore", "memory://stripped.tif"}}, context_)
          .result());

  EXPECT_EQ(dtype_v<uint8_t>, store.dtype());
  EXPECT_EQ(2, store.rank());
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(6, 8));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(2, 8));
  EXPECT_THAT(layout.write_chunk_shape(), ::testing::ElementsAre(2, 8));
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre(0, 1));

  auto expected_array =
      tensorstore::MakeArray<uint8_t>({{0, 1, 2, 3, 4, 5, 6, 7},
                                       {8, 9, 10, 11, 12, 13, 14, 15},
                                       {10, 11, 12, 13, 14, 15, 16, 17},
                                       {18, 19, 20, 21, 22, 23, 24, 25},
                                       {20, 21, 22, 23, 24, 25, 26, 27},
                                       {28, 29, 30, 31, 32, 33, 34, 35}});

  EXPECT_THAT(tensorstore::Read(store).result(), Optional(expected_array));

  // Slice spanning multiple strips.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto slice_view,
      store | tensorstore::Dims(0, 1).SizedInterval({1, 2}, {3, 4}));

  auto expected_slice_array = tensorstore::MakeOffsetArray<uint8_t>(
      {1, 2}, {{10, 11, 12, 13}, {12, 13, 14, 15}, {20, 21, 22, 23}});

  EXPECT_THAT(tensorstore::Read(slice_view).result(),
              Optional(expected_slice_array));
}

TEST_F(TiffDriverTest, ReadFloatTiff) {
  WriteTiffData("float_test.tif", MakeFloatTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "tiff"},
                                     {"kvstore", "memory://float_test.tif"}},
                                    context_)
                      .result());

  EXPECT_EQ(dtype_v<float>, store.dtype());
  EXPECT_EQ(2, store.rank());
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(2, 3));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(layout.write_chunk_shape(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre(0, 1));
  auto expected_array =
      tensorstore::MakeArray<float>({{1.1f, 2.2f, 3.3f}, {4.4f, 5.5f, 6.6f}});

  EXPECT_THAT(tensorstore::Read(store).result(), Optional(expected_array));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto slice_view,
      store | tensorstore::Dims(0, 1).SizedInterval({1, 1}, {1, 2}));

  auto expected_slice_array =
      tensorstore::MakeOffsetArray<float>({1, 1}, {{5.5f, 6.6f}});
  EXPECT_THAT(tensorstore::Read(slice_view).result(), expected_slice_array);
}

TEST_F(TiffDriverTest, ReadMultiChannelTiff) {
  WriteTiffData("multi_channel.tif", MakeMultiChannelTiff());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "tiff"},
                                     {"kvstore", "memory://multi_channel.tif"}},
                                    context_)
                      .result());

  EXPECT_EQ(dtype_v<uint8_t>, store.dtype());
  EXPECT_EQ(3, store.rank());
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0, 0));
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(2, 3, 3));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(1, 1, 3));
  EXPECT_THAT(layout.write_chunk_shape(), ::testing::ElementsAre(1, 1, 3));
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre(0, 1, 2));

  auto expected_array = tensorstore::MakeArray<uint8_t>(
      {{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}},
       {{11, 12, 13}, {12, 13, 14}, {13, 14, 15}}});

  EXPECT_THAT(tensorstore::Read(store).result(), Optional(expected_array));

  // Read single pixel.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto pixel_view, store | tensorstore::Dims(0, 1).IndexSlice({1, 2}));
  auto expected_pixel_array = tensorstore::MakeArray<uint8_t>({13, 14, 15});

  EXPECT_THAT(tensorstore::Read(pixel_view).result(),
              Optional(expected_pixel_array));
}

TEST_F(TiffDriverTest, ReadNonZeroIFD) {
  WriteTiffData("multi_ifd.tif", MakeMultiIFDTiff());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "tiff"},
                                     {"kvstore", "memory://multi_ifd.tif"},
                                     {"tiff", {{"ifd", 1}}}},
                                    context_)
                      .result());

  EXPECT_EQ(dtype_v<uint16_t>, store.dtype());
  EXPECT_EQ(2, store.rank());
  EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(3, 3));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, store.chunk_layout());
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre(0, 1));

  auto expected_array = tensorstore::AllocateArray<uint16_t>(
      {3, 3}, tensorstore::ContiguousLayoutOrder::c, tensorstore::value_init);
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j) expected_array(i, j) = 99;

  EXPECT_THAT(tensorstore::Read(store).result(), Optional(expected_array));
}

}  // namespace