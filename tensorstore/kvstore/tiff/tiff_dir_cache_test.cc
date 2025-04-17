// Copyright 2025 The TensorStore Authors
//
// Licensed under the Apache License, Version .0 (the "License");
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

#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/tiff/tiff_test_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::InlineExecutor;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::GetCache;
using ::tensorstore::internal_tiff_kvstore::TiffDirectoryCache;
using ::tensorstore::internal_tiff_kvstore::testing::TiffBuilder;

TEST(TiffDirectoryCacheTest, ReadSlice) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a small TIFF file with a valid header and IFD
  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(6)  // 6 entries
          // Width and height
          .AddEntry(256, 3, 1, 800)  // ImageWidth = 800
          .AddEntry(257, 3, 1, 600)  // ImageLength = 600
          // Tile info
          .AddEntry(322, 3, 1, 256)  // TileWidth = 256
          .AddEntry(323, 3, 1, 256)  // TileLength = 256
          .AddEntry(324, 4, 1, 128)  // TileOffsets = 128
          .AddEntry(325, 4, 1, 256)  // TileByteCounts = 256
          .EndIfd()                  // No more IFDs
          .PadTo(2048)  // Pad to 2048 bytes (more than kInitialReadBytes)
          .Build();

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "test.tiff");

  // Request with specified range - should read first 1024 bytes
  {
    tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
    request.staleness_bound = absl::InfinitePast();

    ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
    auto* data = lock.data();
    ASSERT_THAT(data, ::testing::NotNull());
    EXPECT_FALSE(data->full_read);

    // Check parsed directories
    EXPECT_EQ(data->directories.size(), 1);
    EXPECT_EQ(data->directories[0].entries.size(), 6);
    EXPECT_EQ(data->image_directories.size(), 1);

    // Check parsed image directory
    EXPECT_EQ(data->image_directories[0].width, 800);
    EXPECT_EQ(data->image_directories[0].height, 600);
    EXPECT_EQ(data->image_directories[0].tile_width, 256);
    EXPECT_EQ(data->image_directories[0].tile_height, 256);
  }
}

TEST(TiffDirectoryCacheTest, ReadFull) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a small TIFF file with a valid header and IFD - similar to above but
  // smaller
  TiffBuilder builder;
  auto tiff_data = builder
                       .StartIfd(5)  // 5 entries
                       // Add strip-based entries
                       .AddEntry(256, 3, 1, 400)  // ImageWidth = 400
                       .AddEntry(257, 3, 1, 300)  // ImageLength = 300
                       .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
                       .AddEntry(273, 4, 1, 128)  // StripOffsets = 128
                       .AddEntry(279, 4, 1, 200)  // StripByteCounts = 200
                       .EndIfd()                  // No more IFDs
                       .PadTo(512)                // Pad to fill data
                       .Build();

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "test.tiff");

  // Request with no specified range - should read entire file
  {
    tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
    request.staleness_bound = absl::InfinitePast();

    ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
    auto* data = lock.data();
    ASSERT_THAT(data, ::testing::NotNull());
    EXPECT_TRUE(data->full_read);

    // Check parsed directories
    EXPECT_EQ(data->directories.size(), 1);
    EXPECT_EQ(data->directories[0].entries.size(), 5);
    EXPECT_EQ(data->image_directories.size(), 1);

    // Check parsed image directory
    EXPECT_EQ(data->image_directories[0].width, 400);
    EXPECT_EQ(data->image_directories[0].height, 300);
    EXPECT_EQ(data->image_directories[0].rows_per_strip, 100);
    EXPECT_EQ(data->image_directories[0].strip_offsets.size(), 1);
    EXPECT_EQ(data->image_directories[0].strip_offsets[0], 128);
    EXPECT_EQ(data->image_directories[0].strip_bytecounts.size(), 1);
    EXPECT_EQ(data->image_directories[0].strip_bytecounts[0], 200);
  }
}

TEST(TiffDirectoryCacheTest, BadIfdFailsParse) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a corrupt TIFF file with invalid IFD
  TiffBuilder builder;
  auto corrupt_tiff = builder
                          .StartIfd(10)  // Claim 10 entries (too many)
                          // Only provide data for 1 entry
                          .AddEntry(1, 1, 1, 0)
                          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "corrupt.tiff",
                                          absl::Cord(corrupt_tiff))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "corrupt.tiff");

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  // Reading should fail due to corrupt IFD
  auto read_result = entry->Read(request).result();
  EXPECT_THAT(read_result.status(), ::testing::Not(::tensorstore::IsOk()));
  EXPECT_TRUE(absl::IsDataLoss(read_result.status()) ||
              absl::IsInvalidArgument(read_result.status()));
}

TEST(TiffDirectoryCacheTest, ExternalArrays_EagerLoad) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file with external array references
  uint32_t strip_offsets_offset = 200;     // Position of external array in file
  uint32_t strip_bytecounts_offset = 216;  // Position of external array in file
  uint32_t strip_offsets[4] = {1000, 2000, 3000, 4000};
  uint32_t strip_bytecounts[4] = {500, 600, 700, 800};

  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(5)  // 5 entries
          // Basic image info
          .AddEntry(256, 3, 1, 800)  // ImageWidth = 800
          .AddEntry(257, 3, 1, 600)  // ImageLength = 600
          .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
          // External arrays
          .AddEntry(273, 4, 4,
                    strip_offsets_offset)  // StripOffsets - external array
          .AddEntry(
              279, 4, 4,
              strip_bytecounts_offset)  // StripByteCounts - external array
          .EndIfd()                     // No more IFDs
          .PadTo(strip_offsets_offset)  // Pad to external array location
          .AddUint32Array({strip_offsets[0], strip_offsets[1], strip_offsets[2],
                           strip_offsets[3]})
          .AddUint32Array({strip_bytecounts[0], strip_bytecounts[1],
                           strip_bytecounts[2], strip_bytecounts[3]})
          .PadTo(4096)  // Pad the file to ensure it's large enough
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "external_arrays.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "external_arrays.tiff");

  // Request to read the TIFF with external arrays
  {
    tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
    request.staleness_bound = absl::InfinitePast();

    ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
    auto* data = lock.data();
    ASSERT_THAT(data, ::testing::NotNull());

    // Check that external arrays were loaded
    EXPECT_EQ(data->image_directories[0].strip_offsets.size(), 4);
    EXPECT_EQ(data->image_directories[0].strip_bytecounts.size(), 4);

    // Verify the external array values were loaded correctly
    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(data->image_directories[0].strip_offsets[i], strip_offsets[i]);
      EXPECT_EQ(data->image_directories[0].strip_bytecounts[i],
                strip_bytecounts[i]);
    }
  }
}

TEST(TiffDirectoryCacheTest, ExternalArrays_BadPointer) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file with an invalid external array reference
  uint32_t invalid_offset = 50000;  // Far beyond our file size

  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(5)  // 5 entries
          // Basic image info
          .AddEntry(256, 3, 1, 800)  // ImageWidth = 800
          .AddEntry(257, 3, 1, 600)  // ImageLength = 600
          .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
          // External strip offsets array with INVALID OFFSET
          .AddEntry(273, 4, 4,
                    invalid_offset)  // StripOffsets - invalid location
          // Valid strip bytecounts
          .AddEntry(279, 4, 1, 500)  // StripByteCounts - inline value
          .EndIfd()                  // No more IFDs
          .PadTo(
              1000)  // Pad to a reasonable size, but less than invalid_offset
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "bad_external_array.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "bad_external_array.tiff");

  // Reading should fail due to invalid external array pointer
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  auto read_result = entry->Read(request).result();
  EXPECT_THAT(read_result.status(), ::testing::Not(::tensorstore::IsOk()));

  std::cout << "Status: " << read_result.status() << std::endl;
  // Should fail with OutOfRange, InvalidArgument, or DataLoss error
  EXPECT_TRUE(absl::IsOutOfRange(read_result.status()) ||
              absl::IsDataLoss(read_result.status()) ||
              absl::IsInvalidArgument(read_result.status()) ||
              absl::IsFailedPrecondition(read_result.status()));
}

// Helper to create a test TIFF file with multiple IFDs
std::string MakeMultiPageTiff() {
  TiffBuilder builder;

  // First IFD at offset 8
  return builder
      .StartIfd(5)  // 5 entries
      // Add strip-based entries for first IFD
      .AddEntry(256, 3, 1, 400)   // ImageWidth = 400
      .AddEntry(257, 3, 1, 300)   // ImageLength = 300
      .AddEntry(278, 3, 1, 100)   // RowsPerStrip = 100
      .AddEntry(273, 4, 1, 1000)  // StripOffsets = 1000
      .AddEntry(279, 4, 1, 200)   // StripByteCounts = 200
      .EndIfd(200)                // Point to second IFD at offset 200
      .PadTo(200)                 // Pad to second IFD offset
      // Second IFD
      .StartIfd(6)  // 6 entries
      // Add tile-based entries for second IFD
      .AddEntry(256, 3, 1, 800)   // ImageWidth = 800
      .AddEntry(257, 3, 1, 600)   // ImageLength = 600
      .AddEntry(322, 3, 1, 256)   // TileWidth = 256
      .AddEntry(323, 3, 1, 256)   // TileLength = 256
      .AddEntry(324, 4, 1, 2000)  // TileOffsets
      .AddEntry(325, 4, 1, 300)   // TileByteCounts
      .EndIfd()                   // No more IFDs
      .PadTo(3000)                // Pad file to cover all offsets
      .Build();
}

TEST(TiffDirectoryCacheMultiIfdTest, ReadAndVerifyIFDs) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "multi_ifd.tiff",
                                          absl::Cord(MakeMultiPageTiff()))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "multi_ifd.tiff");

  // Request to read the TIFF with multiple IFDs
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Verify we have two IFDs
  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  // Check first IFD (strip-based)
  const auto& ifd1 = data->directories[0];
  const auto& img1 = data->image_directories[0];
  EXPECT_EQ(ifd1.entries.size(), 5);
  EXPECT_EQ(img1.width, 400);
  EXPECT_EQ(img1.height, 300);
  EXPECT_EQ(img1.rows_per_strip, 100);
  EXPECT_EQ(img1.strip_offsets.size(), 1);
  EXPECT_EQ(img1.strip_offsets[0], 1000);
  EXPECT_EQ(img1.strip_bytecounts[0], 200);

  // Check second IFD (tile-based)
  const auto& ifd2 = data->directories[1];
  const auto& img2 = data->image_directories[1];
  EXPECT_EQ(ifd2.entries.size(), 6);
  EXPECT_EQ(img2.width, 800);
  EXPECT_EQ(img2.height, 600);
  EXPECT_EQ(img2.tile_width, 256);
  EXPECT_EQ(img2.tile_height, 256);
  EXPECT_EQ(img2.tile_offsets.size(), 1);
  EXPECT_EQ(img2.tile_offsets[0], 2000);

  // Since our test file is larger than kInitialReadBytes (1024),
  // it should be not be fully read in one shot
  EXPECT_FALSE(data->full_read);
}

TEST(TiffDirectoryCacheMultiIfdTest, ReadLargeMultiPageTiff) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file larger than kInitialReadBytes
  TiffBuilder builder;
  auto tiff_data =
      builder
          // First IFD
          .StartIfd(5)               // 5 entries
          .AddEntry(256, 3, 1, 400)  // ImageWidth = 400
          .AddEntry(257, 3, 1, 300)  // ImageLength = 300
          .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
          .AddEntry(273, 4, 1,
                    1024)  // StripOffsets = 1024 (just after initial read)
          .AddEntry(279, 4, 1, 200)  // StripByteCounts = 200
          .EndIfd(2048)  // Point to second IFD at offset 2048 (well beyond
                         // initial read)
          .PadTo(2048)   // Pad to second IFD offset
          // Second IFD
          .StartIfd(6)                // 6 entries
          .AddEntry(256, 3, 1, 800)   // ImageWidth = 800
          .AddEntry(257, 3, 1, 600)   // ImageLength = 600
          .AddEntry(322, 3, 1, 256)   // TileWidth = 256
          .AddEntry(323, 3, 1, 256)   // TileLength = 256
          .AddEntry(324, 4, 1, 3000)  // TileOffsets
          .AddEntry(325, 4, 1, 300)   // TileByteCounts
          .EndIfd()                   // No more IFDs
          .PadTo(4096)                // Pad file to cover all offsets
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "large_multi_ifd.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "large_multi_ifd.tiff");

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Verify we have two IFDs
  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  // Verify both IFDs were correctly parsed despite being in different chunks
  EXPECT_EQ(data->image_directories[0].width, 400);
  EXPECT_EQ(data->image_directories[1].width, 800);
}

TEST(TiffDirectoryCacheMultiIfdTest, ExternalArraysMultiIfdTest) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Build a TIFF file with two IFDs, each referencing external arrays
  std::vector<uint32_t> offsets1 = {1000, 2000, 3000, 4000};
  std::vector<uint32_t> bytecounts1 = {50, 60, 70, 80};
  std::vector<uint32_t> offsets2 = {5000, 5004, 5008, 5012};
  std::vector<uint32_t> bytecounts2 = {100, 200, 300, 400};

  TiffBuilder builder;
  auto tiff_data =
      builder
          // First IFD with external arrays
          .StartIfd(5)               // 5 entries
          .AddEntry(256, 3, 1, 400)  // ImageWidth
          .AddEntry(257, 3, 1, 300)  // ImageLength
          .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
          .AddEntry(273, 4, 4,
                    512)  // StripOffsets array (points to offset 512)
          .AddEntry(279, 4, 4,
                    528)  // StripByteCounts array (points to offset 528)
          .EndIfd(600)    // Second IFD offset at 600
          .PadTo(512)     // Pad to 512
          // External arrays for first IFD
          .AddUint32Array(offsets1)
          .AddUint32Array(bytecounts1)
          .PadTo(600)  // Pad to second IFD offset
          // Second IFD with external arrays
          .StartIfd(6)               // 6 entries
          .AddEntry(256, 3, 1, 800)  // ImageWidth
          .AddEntry(257, 3, 1, 600)  // ImageLength
          .AddEntry(322, 3, 1, 256)  // TileWidth
          .AddEntry(323, 3, 1, 256)  // TileLength
          .AddEntry(324, 4, 4, 700)  // TileOffsets array (offset 700)
          .AddEntry(325, 4, 4, 716)  // TileByteCounts array (offset 716)
          .EndIfd()                  // No more IFDs
          .PadTo(700)                // Pad to external arrays for second IFD
          .AddUint32Array(offsets2)
          .AddUint32Array(bytecounts2)
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "multi_ifd_external.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "multi_ifd_external.tiff");

  // Read back with TiffDirectoryCache
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Expect two IFDs
  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  // Check external arrays in IFD #1
  EXPECT_EQ(data->image_directories[0].strip_offsets.size(), 4);
  EXPECT_EQ(data->image_directories[0].strip_bytecounts.size(), 4);

  // Check external arrays in IFD #2
  // (Tile offsets and bytecounts are stored, but the key is that they got
  // parsed)
  EXPECT_EQ(data->image_directories[1].tile_offsets.size(), 4);
  EXPECT_EQ(data->image_directories[1].tile_bytecounts.size(), 4);
}

TEST(TiffDirectoryCacheTest, ExternalArrays_Uint16Arrays) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file with uint16_t external arrays (BitsPerSample and
  // SampleFormat)
  uint32_t bits_per_sample_offset = 200;
  uint32_t sample_format_offset = 212;
  std::vector<uint16_t> bits_values = {8, 8, 8};  // 8 bits per channel
  std::vector<uint16_t> sample_format_values = {1, 1,
                                                1};  // 1 = unsigned integer

  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(8)  // 8 entries
          // Basic image info
          .AddEntry(256, 3, 1, 800)  // ImageWidth = 800
          .AddEntry(257, 3, 1, 600)  // ImageLength = 600
          .AddEntry(277, 3, 1, 3)    // SamplesPerPixel = 3 (RGB)
          .AddEntry(278, 3, 1, 100)  // RowsPerStrip = 100
          // External arrays
          .AddEntry(258, 3, 3,
                    bits_per_sample_offset)  // BitsPerSample - external array
          .AddEntry(339, 3, 3,
                    sample_format_offset)  // SampleFormat - external array
          // Required entries
          .AddEntry(273, 4, 1, 1000)      // StripOffsets = 1000
          .AddEntry(279, 4, 1, 30000)     // StripByteCounts = 30000
          .EndIfd()                       // No more IFDs
          .PadTo(bits_per_sample_offset)  // Pad to BitsPerSample external array
                                          // location
          .AddUint16Array(bits_values)    // Write BitsPerSample external array
          .PadTo(sample_format_offset)    // Make sure we're at the
                                          // sample_format_offset
          .AddUint16Array(
              sample_format_values)  // Write SampleFormat external array
          .PadTo(2048)               // Pad the file to ensure it's large enough
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "uint16_arrays.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "uint16_arrays.tiff");

  // Request to read the TIFF with external uint16_t arrays
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Check that the uint16_t external arrays were loaded properly
  const auto& img_dir = data->image_directories[0];

  // Check SamplesPerPixel
  EXPECT_EQ(img_dir.samples_per_pixel, 3);

  // Check RowsPerStrip
  EXPECT_EQ(img_dir.rows_per_strip, 100);

  // Check BitsPerSample array
  ASSERT_EQ(img_dir.bits_per_sample.size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(img_dir.bits_per_sample[i], bits_values[i]);
  }

  // Check SampleFormat array
  ASSERT_EQ(img_dir.sample_format.size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(img_dir.sample_format[i], sample_format_values[i]);
  }
}

// Add a comprehensive test that checks all supported TIFF tags
TEST(TiffDirectoryCacheTest, ComprehensiveTiffTagsTest) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file with all supported tags
  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(11)  // 11 entries (all standard tags we support)
          // Add all standard tags with their test values
          .AddEntry(256, 3, 1, 1024)  // ImageWidth = 1024
          .AddEntry(257, 3, 1, 768)   // ImageLength = 768
          .AddEntry(258, 3, 1, 16)  // BitsPerSample = 16 (single value, inline)
          .AddEntry(259, 3, 1, 1)   // Compression = 1 (none)
          .AddEntry(262, 3, 1, 2)   // PhotometricInterpretation = 2 (RGB)
          .AddEntry(277, 3, 1, 1)   // SamplesPerPixel = 1
          .AddEntry(278, 3, 1, 128)    // RowsPerStrip = 128
          .AddEntry(273, 4, 1, 1000)   // StripOffsets = 1000
          .AddEntry(279, 4, 1, 65536)  // StripByteCounts = 65536
          .AddEntry(284, 3, 1, 1)      // PlanarConfiguration = 1 (chunky)
          .AddEntry(339, 3, 1, 1)      // SampleFormat = 1 (unsigned)
          .EndIfd()                    // No more IFDs
          .PadTo(2048)  // Pad the file to ensure it's large enough
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "comprehensive_tags.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "comprehensive_tags.tiff");

  // Read the TIFF
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Verify all tags were parsed correctly
  const auto& img_dir = data->image_directories[0];
  EXPECT_EQ(img_dir.width, 1024);
  EXPECT_EQ(img_dir.height, 768);
  ASSERT_EQ(img_dir.bits_per_sample.size(), 1);
  EXPECT_EQ(img_dir.bits_per_sample[0], 16);
  EXPECT_EQ(img_dir.compression, 1);  // None
  EXPECT_EQ(img_dir.photometric, 2);  // RGB
  EXPECT_EQ(img_dir.samples_per_pixel, 1);
  EXPECT_EQ(img_dir.rows_per_strip, 128);
  ASSERT_EQ(img_dir.strip_offsets.size(), 1);
  EXPECT_EQ(img_dir.strip_offsets[0], 1000);
  ASSERT_EQ(img_dir.strip_bytecounts.size(), 1);
  EXPECT_EQ(img_dir.strip_bytecounts[0], 65536);
  EXPECT_EQ(img_dir.planar_config, 1);  // Chunky
  ASSERT_EQ(img_dir.sample_format.size(), 1);
  EXPECT_EQ(img_dir.sample_format[0], 1);  // Unsigned integer
}

// Add a test for a tiled TIFF with all supported tags
TEST(TiffDirectoryCacheTest, TiledTiffWithAllTags) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a tiled TIFF file with all supported tags
  TiffBuilder builder;
  auto tiff_data =
      builder
          .StartIfd(
              12)  // 12 entries (all standard tags we support for tiled TIFF)
          // Add all standard tags with their test values for a tiled TIFF
          .AddEntry(256, 3, 1, 2048)  // ImageWidth = 2048
          .AddEntry(257, 3, 1, 2048)  // ImageLength = 2048
          .AddEntry(258, 3, 1, 32)    // BitsPerSample = 32
          .AddEntry(259, 3, 1, 8)     // Compression = 8 (Deflate)
          .AddEntry(262, 3, 1,
                    1)  // PhotometricInterpretation = 1 (BlackIsZero)
          .AddEntry(277, 3, 1, 1)  // SamplesPerPixel = 1
          .AddEntry(284, 3, 1, 1)  // PlanarConfiguration = 1 (chunky)
          .AddEntry(339, 3, 1, 3)  // SampleFormat = 3 (IEEE float)
          // Tile-specific tags
          .AddEntry(322, 3, 1, 256)    // TileWidth = 256
          .AddEntry(323, 3, 1, 256)    // TileLength = 256
          .AddEntry(324, 4, 1, 1000)   // TileOffsets = 1000
          .AddEntry(325, 4, 1, 10000)  // TileByteCounts = 10000
          .EndIfd()                    // No more IFDs
          .PadTo(2048)  // Pad the file to ensure it's large enough
          .Build();

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "tiled_tiff_all_tags.tiff",
                                          absl::Cord(tiff_data))
                  .result(),
              ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver,
                                                InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "tiled_tiff_all_tags.tiff");

  // Read the TIFF
  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  // Verify all tags were parsed correctly
  const auto& img_dir = data->image_directories[0];

  // Basic image properties
  EXPECT_EQ(img_dir.width, 2048);
  EXPECT_EQ(img_dir.height, 2048);
  ASSERT_EQ(img_dir.bits_per_sample.size(), 1);
  EXPECT_EQ(img_dir.bits_per_sample[0], 32);
  EXPECT_EQ(img_dir.compression, 8);  // Deflate
  EXPECT_EQ(img_dir.photometric, 1);  // BlackIsZero
  EXPECT_EQ(img_dir.samples_per_pixel, 1);
  EXPECT_EQ(img_dir.planar_config, 1);  // Chunky
  ASSERT_EQ(img_dir.sample_format.size(), 1);
  EXPECT_EQ(img_dir.sample_format[0], 3);  // IEEE float

  // Tile-specific properties
  EXPECT_EQ(img_dir.tile_width, 256);
  EXPECT_EQ(img_dir.tile_height, 256);
  ASSERT_EQ(img_dir.tile_offsets.size(), 1);
  EXPECT_EQ(img_dir.tile_offsets[0], 1000);
  ASSERT_EQ(img_dir.tile_bytecounts.size(), 1);
  EXPECT_EQ(img_dir.tile_bytecounts[0], 10000);
}

}  // namespace