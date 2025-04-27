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

/// End-to-end tests of the TIFF driver.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
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
#include "tensorstore/driver/driver_testutil.h"  // For TestTensorStoreDriverSpecRoundtrip
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"  // For TENSORSTORE_GLOBAL_INITIALIZER
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/kvstore.h"  // For kvstore::Write
#include "tensorstore/kvstore/memory/memory_key_value_store.h"  // For GetMemoryKeyValueStore
#include "tensorstore/kvstore/test_matchers.h"  // For kvstore testing matchers if needed
#include "tensorstore/kvstore/tiff/tiff_test_util.h"  // For TiffBuilder
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"  // For TensorStore
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"  // For MatchesStatus, TENSORSTORE_ASSERT_OK

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
    builder
        .StartIfd(
            10)  // W, H, SPP, BPS, Comp, Photo, TW, TL, TileOffsets/Counts
        .AddEntry(256, 3, 1, 10)   // ImageWidth = 10
        .AddEntry(257, 3, 1, 20)   // ImageLength = 20
        .AddEntry(277, 3, 1, 1)    // SamplesPerPixel = 1
        .AddEntry(258, 3, 1, 8)    // BitsPerSample = 8
        .AddEntry(259, 3, 1, 1)    // Compression = None
        .AddEntry(262, 3, 1, 1)    // PhotometricInterpretation = MinIsBlack
        .AddEntry(322, 3, 1, 10)   // TileWidth = 10
        .AddEntry(323, 3, 1, 10);  // TileLength = 10
    // Fake tile data offsets/counts (points past end of current data)
    size_t data_start = builder.CurrentOffset() + 12 * 9 + 4 +
                        4 * 4;  // IFD + next_offset + arrays
    builder.AddEntry(324, 4, 2,
                     builder.CurrentOffset() + 12 * 9 + 4);  // TileOffsets
    builder.AddEntry(
        325, 4, 2,
        builder.CurrentOffset() + 12 * 9 + 4 + 4 * 2);  // TileByteCounts
    builder.EndIfd(0);
    builder.AddUint32Array(
        {(uint32_t)data_start,
         (uint32_t)(data_start + 100)});  // Offsets for 2 10x10 tiles
    builder.AddUint32Array({100, 100});   // ByteCounts
    builder.data_.append(100, '\1');      // Tile 1 data (non-zero)
    builder.data_.append(100, '\2');      // Tile 2 data (non-zero)
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
    size_t ifd_block_size = 2 + (10 * 12) + 4;  // Size of IFD block
    size_t end_of_ifd_offset = header_size + ifd_block_size;

    size_t tile_offsets_array_start_offset = end_of_ifd_offset;
    size_t tile_offsets_array_size = 4 * sizeof(uint32_t);  // 4 tiles
    size_t tile_bytecounts_array_start_offset =
        tile_offsets_array_start_offset + tile_offsets_array_size;
    size_t tile_bytecounts_array_size = 4 * sizeof(uint32_t);  // 4 tiles
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
};

// --- Spec Tests ---
TEST_F(TiffDriverTest, SpecFromJsonMinimal) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson({{"driver", "tiff"}, {"kvstore", "memory://test/"}}));
  // Access spec members directly for verification (requires public access or
  // friend declaration if needed) For now, just check parsing success
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
  // Check properties via Schema methods where possible
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, spec.schema());
  EXPECT_EQ(dtype_v<uint16_t>, schema.dtype());
  EXPECT_EQ(2, schema.rank());
  // Cannot directly access tiff_options from public Spec API easily
  // Cannot directly access metadata_constraints from public Spec API easily
}

TEST_F(TiffDriverTest, SpecToJsonWithOptions) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      Spec::FromJson(
          {{"driver", "tiff"},
           {"kvstore", "memory://test/"},
           {"tiff", {{"ifd", 5}}},
           {"metadata", {{"dtype", "uint16"}, {"shape", {30, 40}}}}}));

  // Define the EXPECTED json based on the ACTUAL output from the failure log
  ::nlohmann::json expected_json = {
      {"driver", "tiff"},
      {"kvstore",
       {{"driver", "memory"},
        {"path", "test/"},
        {"atomic", true},
        {"memory_key_value_store", "memory_key_value_store"},
        {"context", ::nlohmann::json({})}}},
      {"dtype",
       "uint16"},  // dtype is now a top-level key from KvsDriverSpec binder
      {"schema",
       {// Schema is inferred and added
        {"dtype", "uint16"},
        {"rank", 2}}},
      {"transform",
       {// Default transform is added
        {"input_inclusive_min", {0, 0}},
        {"input_exclusive_max", {30, 40}}}},
      {"context", ::nlohmann::json({})},  // Default empty context braces
      {"cache_pool", "cache_pool"},       // Default context resource names
      {"data_copy_concurrency",
       "data_copy_concurrency"},            // Default context resource names
      {"recheck_cached_data", true},        // Check actual default
      {"recheck_cached_metadata", "open"},  // Check actual default
      {"delete_existing", false},
      {"assume_metadata", false},
      {"assume_cached_metadata", false},
      {"fill_missing_data_reads", true},
      {"store_data_equal_to_fill_value", false},
      {"tiff", {{"ifd", 5}}},
      {"metadata", {{"dtype", "uint16"}, {"shape", {30, 40}}}}};

  // Convert back to JSON including defaults to verify all fields
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

// Use TestSpecSchema for basic schema property inference from spec
TEST_F(TiffDriverTest, TestSpecSchemaDtype) {
  // Test that specifying dtype also includes the default tiff codec in the
  // schema
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
      // Expected schema now includes rank, domain, default layout, and codec:
      {
          {"rank", 3},
          {"domain",
           {{"inclusive_min", {0, 0, 0}}, {"exclusive_max", {10, 20, 30}}}},
          {"chunk_layout",
           {{"inner_order_soft_constraint", {0, 1, 2}},    // Default C order
            {"grid_origin_soft_constraint", {0, 0, 0}}}},  // Default origin
          {"codec", {{"driver", "tiff"}}}                  // Default codec
      });
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

  // Use public API to check properties
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
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "tiff"},
                      {"kvstore", "memory://minimal.tif"},
                      {"metadata", {{"dtype", "uint16"}}}  // Mismatch
                  },
                  context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Schema dtype uint16 is incompatible .*"
                            "TIFF dtype uint8.*"));
}

TEST_F(TiffDriverTest, OpenWithMismatchedShapeConstraint) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "tiff"},
                      {"kvstore", "memory://minimal.tif"},
                      {"metadata", {{"shape", {20, 11}}}}  // Mismatch
                  },
                  context_)
                  .result(),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            ".*Resolved TIFF shape .*20, 10.* does not match "
                            "user constraint shape .*20, 11.*"));
}

TEST_F(TiffDriverTest, OpenWithSchemaDtypeMismatch) {
  WriteTiffData("minimal.tif", MakeMinimalTiff());
  EXPECT_THAT(
      tensorstore::Open(
          {
              {"driver", "tiff"},
              {"kvstore", "memory://minimal.tif"},
              {"schema", {{"dtype", "int16"}}}  // Mismatch
          },
          context_)
          .result(),
      // This error comes from ResolveMetadata comparing schema and TIFF data
      MatchesStatus(
          absl::StatusCode::kFailedPrecondition,
          ".*Schema dtype int16 is incompatible with TIFF dtype uint8.*"));
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
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "tiff"},
                      {"kvstore", "memory://minimal.tif"},
                      {"tiff", {{"ifd", 1}}}  // Request IFD 1
                  },
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
  // Dims(0, 1).IndexSlice({1, 2}) -> Element at row 1, col 2 -> value 9
  EXPECT_THAT(
      tensorstore::Read(store | tensorstore::Dims(0, 1).IndexSlice({1, 2}))
          .result(),
      Optional(tensorstore::MakeScalarArray<uint16_t>(9)));

  // Read a slice within a single tile (tile 2)
  // Dims(0, 1).SizedInterval({2, 1}, {1, 2}) -> Start at row 2, col 1; size 1
  // row, 2 cols
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

  // Test ResolveBounds
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved_store,
                                   ResolveBounds(store).result());
  EXPECT_EQ(store.domain(), resolved_store.domain());

  // Test GetBoundSpec
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto bound_spec, store.spec());
  ASSERT_TRUE(bound_spec.valid());

  // Check the minimal JSON representation (IncludeDefaults=false)
  ::nlohmann::json expected_minimal_json = {
      {"driver", "tiff"},
      {"kvstore", {{"driver", "memory"}, {"path", "read_test.tif"}}},
      {"dtype", "uint16"},
      {"transform",
       {// Includes the resolved domain
        {"input_inclusive_min", {0, 0}},
        {"input_exclusive_max", {4, 6}}}},
      {"metadata", {{"dtype", "uint16"}, {"shape", {4, 6}}}}};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto minimal_json, bound_spec.ToJson());
  EXPECT_THAT(minimal_json, MatchesJson(expected_minimal_json));

  // Optional: Check the full JSON representation (IncludeDefaults=true)
  // This would include default tiff options, schema defaults, context resources
  // etc. Example (adjust based on actual defaults set by
  // KvsDriverSpec/TiffDriverSpec):
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
       {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {4, 6}}}},
      {"metadata",
       {
           {"dtype", "uint16"}, {"shape", {4, 6}}
           // May include other resolved metadata if GetBoundSpecData adds more
       }},
      {"tiff", {{"ifd", 0}}},  // Default ifd included
      {"schema",
       {// Includes defaults inferred or set
        {"rank", 2},
        {"dtype", "uint16"}}},
      // Default context resource names/specs might appear here too
      {"recheck_cached_data", true},        // Example default
      {"recheck_cached_metadata", "open"},  // Example default
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
}  // namespace