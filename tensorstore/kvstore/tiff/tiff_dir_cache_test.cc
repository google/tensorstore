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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::InlineExecutor;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::GetCache;
using ::tensorstore::internal_tiff_kvstore::TiffDirectoryCache;


TEST(TiffDirectoryCacheTest, ReadSlice) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a small TIFF file with a valid header and IFD
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 5 entries
  tiff_data.push_back(6); tiff_data.push_back(0);  // 5 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Width and height
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  
  // Tile info
  AddEntry(322, 3, 1, 256);  // TileWidth = 256
  AddEntry(323, 3, 1, 256);  // TileLength = 256
  AddEntry(324, 4, 1, 128);  // TileOffsets = 128
  AddEntry(325, 4, 1, 256);  // TileByteCounts = 256
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad to 2048 bytes (more than kInitialReadBytes)
  while (tiff_data.size() < 2048) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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

  // Create a small TIFF file with a valid header and IFD - similar to above but smaller
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                              // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 5 entries
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Add strip-based entries 
  AddEntry(256, 3, 1, 400);  // ImageWidth = 400
  AddEntry(257, 3, 1, 300);  // ImageLength = 300
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  AddEntry(273, 4, 1, 128);  // StripOffsets = 128
  AddEntry(279, 4, 1, 200);  // StripByteCounts = 200
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad to fill data
  while (tiff_data.size() < 512) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string corrupt_tiff;
  
  // Valid TIFF header
  corrupt_tiff += "II";                              // Little endian
  corrupt_tiff.push_back(42); corrupt_tiff.push_back(0);  // Magic number
  corrupt_tiff.push_back(8); corrupt_tiff.push_back(0);   // IFD offset (8)
  corrupt_tiff.push_back(0); corrupt_tiff.push_back(0);
  
  // Corrupt IFD - claim 10 entries but only provide data for 1
  corrupt_tiff.push_back(10); corrupt_tiff.push_back(0);  // 10 entries (too many)
  
  // Only one entry (not enough data for 10)
  corrupt_tiff.push_back(1); corrupt_tiff.push_back(1);   // tag
  corrupt_tiff.push_back(1); corrupt_tiff.push_back(0);   // type
  corrupt_tiff.push_back(1); corrupt_tiff.push_back(0);   // count
  corrupt_tiff.push_back(0); corrupt_tiff.push_back(0);
  corrupt_tiff.push_back(0); corrupt_tiff.push_back(0);   // value
  corrupt_tiff.push_back(0); corrupt_tiff.push_back(0);

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "corrupt.tiff", absl::Cord(corrupt_tiff))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 5 entries
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Basic image info
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  
  // External strip offsets array (4 strips)
  uint32_t strip_offsets_offset = 200; // Position of external array in file
  AddEntry(273, 4, 4, strip_offsets_offset);  // StripOffsets - points to external array
  
  // External strip bytecounts array (4 strips)
  uint32_t strip_bytecounts_offset = 216; // Position of external array in file
  AddEntry(279, 4, 4, strip_bytecounts_offset);  // StripByteCounts - points to external array
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad to 200 bytes to reach strip_offsets_offset
  while (tiff_data.size() < strip_offsets_offset) {
    tiff_data.push_back('X');
  }
  
  // Write the strip offsets external array (4 strips)
  uint32_t strip_offsets[4] = {1000, 2000, 3000, 4000};
  for (uint32_t offset : strip_offsets) {
    tiff_data.push_back(offset & 0xFF);
    tiff_data.push_back((offset >> 8) & 0xFF);
    tiff_data.push_back((offset >> 16) & 0xFF);
    tiff_data.push_back((offset >> 24) & 0xFF);
  }
  
  // Write the strip bytecounts external array (4 strips)
  uint32_t strip_bytecounts[4] = {500, 600, 700, 800};
  for (uint32_t bytecount : strip_bytecounts) {
    tiff_data.push_back(bytecount & 0xFF);
    tiff_data.push_back((bytecount >> 8) & 0xFF);
    tiff_data.push_back((bytecount >> 16) & 0xFF);
    tiff_data.push_back((bytecount >> 24) & 0xFF);
  }
  
  // Pad the file to ensure it's large enough
  while (tiff_data.size() < 4096) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "external_arrays.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
      EXPECT_EQ(data->image_directories[0].strip_bytecounts[i], strip_bytecounts[i]);
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 5 entries
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Basic image info
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  
  // External strip offsets array with INVALID OFFSET - points beyond file end
  uint32_t invalid_offset = 50000;  // Far beyond our file size
  AddEntry(273, 4, 4, invalid_offset);  // StripOffsets - points to invalid location
  
  // Valid strip bytecounts
  AddEntry(279, 4, 1, 500);  // StripByteCounts - inline value
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad the file to a reasonable size, but less than invalid_offset
  while (tiff_data.size() < 1000) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "bad_external_array.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };

  // First IFD at offset 8
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  
  // Add strip-based entries for first IFD
  AddEntry(256, 3, 1, 400);  // ImageWidth = 400
  AddEntry(257, 3, 1, 300);  // ImageLength = 300
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  AddEntry(273, 4, 1, 1000);  // StripOffsets = 1000
  AddEntry(279, 4, 1, 200);  // StripByteCounts = 200
  
  // Point to second IFD at offset 200
  tiff_data.push_back(200); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad to second IFD offset
  while (tiff_data.size() < 200) {
    tiff_data.push_back('X');
  }
  
  // Second IFD
  tiff_data.push_back(6); tiff_data.push_back(0);  // 6 entries
  
  // Add tile-based entries for second IFD
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  AddEntry(322, 3, 1, 256);  // TileWidth = 256
  AddEntry(323, 3, 1, 256);  // TileLength = 256
  AddEntry(324, 4, 1, 2000);  // TileOffsets
  AddEntry(325, 4, 1, 300);   // TileByteCounts (needed for tile-based IFD)
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad file to cover all offsets
  while (tiff_data.size() < 3000) {
    tiff_data.push_back('X');
  }
  
  return tiff_data;
}

TEST(TiffDirectoryCacheMultiIfdTest, ReadAndVerifyIFDs) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "multi_ifd.tiff", 
                                 absl::Cord(MakeMultiPageTiff()))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };

  // First IFD
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  AddEntry(256, 3, 1, 400);  // ImageWidth = 400 
  AddEntry(257, 3, 1, 300);  // ImageLength = 300
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  AddEntry(273, 4, 1, 1024); // StripOffsets = 1024 (just after initial read)
  AddEntry(279, 4, 1, 200);  // StripByteCounts = 200
  
  // Point to second IFD at offset 2048 (well beyond initial read)
  tiff_data.push_back(0x00); tiff_data.push_back(0x08);
  tiff_data.push_back(0x00); tiff_data.push_back(0x00);
  
  // Pad to second IFD offset
  while (tiff_data.size() < 2048) {
    tiff_data.push_back('X');
  }
  
  // Second IFD
  tiff_data.push_back(6); tiff_data.push_back(0);  // 6 entries
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  AddEntry(322, 3, 1, 256);  // TileWidth = 256
  AddEntry(323, 3, 1, 256);  // TileLength = 256
  AddEntry(324, 4, 1, 3000); // TileOffsets
  AddEntry(325, 4, 1, 300);  // TileByteCounts (needed for tile-based IFD)
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad file to cover all offsets
  while (tiff_data.size() < 4096) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "large_multi_ifd.tiff", 
                                 absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  tiff_data += "II";                  // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // First IFD offset
  tiff_data.push_back(0); tiff_data.push_back(0);

  auto AddEntry = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };

  // First IFD with external arrays
  tiff_data.push_back(5); tiff_data.push_back(0);  // 5 entries
  AddEntry(256, 3, 1, 400);  // ImageWidth
  AddEntry(257, 3, 1, 300);  // ImageLength
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  AddEntry(273, 4, 4, 512);  // StripOffsets array (points to offset 512)
  AddEntry(279, 4, 4, 528);  // StripByteCounts array (points to offset 528)

  // Second IFD offset at 600
  tiff_data.push_back(0x58); tiff_data.push_back(0x02);
  tiff_data.push_back(0x00); tiff_data.push_back(0x00);

  // Pad to 512
  while (tiff_data.size() < 512) tiff_data.push_back('X');

  // External arrays for first IFD (4 entries each)
  uint32_t offsets1[4] = {1000, 2000, 3000, 4000};
  for (uint32_t val : offsets1) {
    for (int i = 0; i < 4; i++) {
      tiff_data.push_back((val >> (8 * i)) & 0xFF);
    }
  }
  uint32_t bytecounts1[4] = {50, 60, 70, 80};
  for (uint32_t val : bytecounts1) {
    for (int i = 0; i < 4; i++) {
      tiff_data.push_back((val >> (8 * i)) & 0xFF);
    }
  }

  // Pad to second IFD offset (600)
  while (tiff_data.size() < 600) tiff_data.push_back('X');

  // Second IFD with external arrays
  tiff_data.push_back(6); tiff_data.push_back(0);  // 6 entries
  AddEntry(256, 3, 1, 800);  // ImageWidth
  AddEntry(257, 3, 1, 600);  // ImageLength
  AddEntry(322, 3, 1, 256);  // TileWidth
  AddEntry(323, 3, 1, 256);  // TileLength
  AddEntry(324, 4, 4, 700);  // TileOffsets array (offset 700)
  AddEntry(325, 4, 4, 716);  // TileByteCounts array (offset 716)
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);

  // Pad to external arrays for second IFD
  while (tiff_data.size() < 700) tiff_data.push_back('X');
  uint32_t offsets2[4] = {5000, 5004, 5008, 5012};
  for (auto val : offsets2) {
    for (int i = 0; i < 4; i++) {
      tiff_data.push_back((val >> (8 * i)) & 0xFF);
    }
  }
  uint32_t bytecounts2[4] = {100, 200, 300, 400};
  for (auto val : bytecounts2) {
    for (int i = 0; i < 4; i++) {
      tiff_data.push_back((val >> (8 * i)) & 0xFF);
    }
  }

  // Write the file
  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "multi_ifd_external.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  // Read back with TiffDirectoryCache
  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
  });
  auto entry = GetCacheEntry(cache, "multi_ifd_external.tiff");
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
  // (Tile offsets and bytecounts are stored, but the key is that they got parsed)
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

  // Create a TIFF file with uint16_t external arrays (BitsPerSample and SampleFormat)
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 8 entries
  tiff_data.push_back(8); tiff_data.push_back(0);  // 8 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Basic image info
  AddEntry(256, 3, 1, 800);  // ImageWidth = 800
  AddEntry(257, 3, 1, 600);  // ImageLength = 600
  AddEntry(277, 3, 1, 3);    // SamplesPerPixel = 3 (RGB)
  AddEntry(278, 3, 1, 100);  // RowsPerStrip = 100
  
  // External BitsPerSample array (3 values for RGB)
  uint32_t bits_per_sample_offset = 200;
  AddEntry(258, 3, 3, bits_per_sample_offset);  // BitsPerSample - external array
  
  // External SampleFormat array (3 values for RGB)
  uint32_t sample_format_offset = 212;
  AddEntry(339, 3, 3, sample_format_offset);  // SampleFormat - external array
  
  // Add a StripOffsets and StripByteCounts entry to make this a valid TIFF
  AddEntry(273, 4, 1, 1000);  // StripOffsets = 1000
  AddEntry(279, 4, 1, 30000); // StripByteCounts = 30000
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad to BitsPerSample external array location
  while (tiff_data.size() < bits_per_sample_offset) {
    tiff_data.push_back('X');
  }
  
  // Write BitsPerSample external array - 3 uint16_t values for RGB
  uint16_t bits_values[3] = {8, 8, 8};  // 8 bits per channel
  for (uint16_t val : bits_values) {
    tiff_data.push_back(val & 0xFF);
    tiff_data.push_back((val >> 8) & 0xFF);
  }
  
  // Make sure we're at the sample_format_offset
  while (tiff_data.size() < sample_format_offset) {
    tiff_data.push_back('X');
  }
  
  // Write SampleFormat external array - 3 uint16_t values for RGB
  uint16_t sample_format_values[3] = {1, 1, 1};  // 1 = unsigned integer
  for (uint16_t val : sample_format_values) {
    tiff_data.push_back(val & 0xFF);
    tiff_data.push_back((val >> 8) & 0xFF);
  }
  
  // Pad the file to ensure it's large enough
  while (tiff_data.size() < 2048) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "uint16_arrays.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 11 entries (all standard tags we support)
  tiff_data.push_back(11); tiff_data.push_back(0);  // 11 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Add all standard tags with their test values
  AddEntry(256, 3, 1, 1024);         // ImageWidth = 1024
  AddEntry(257, 3, 1, 768);          // ImageLength = 768
  AddEntry(258, 3, 1, 16);           // BitsPerSample = 16 (single value, inline)
  AddEntry(259, 3, 1, 1);            // Compression = 1 (none)
  AddEntry(262, 3, 1, 2);            // PhotometricInterpretation = 2 (RGB)
  AddEntry(277, 3, 1, 1);            // SamplesPerPixel = 1
  AddEntry(278, 3, 1, 128);          // RowsPerStrip = 128
  AddEntry(273, 4, 1, 1000);         // StripOffsets = 1000
  AddEntry(279, 4, 1, 65536);        // StripByteCounts = 65536
  AddEntry(284, 3, 1, 1);            // PlanarConfiguration = 1 (chunky)
  AddEntry(339, 3, 1, 1);            // SampleFormat = 1 (unsigned)
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad the file to ensure it's large enough
  while (tiff_data.size() < 2048) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "comprehensive_tags.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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
  std::string tiff_data;
  
  // TIFF header (8 bytes)
  tiff_data += "II";                           // Little endian
  tiff_data.push_back(42); tiff_data.push_back(0);  // Magic number
  tiff_data.push_back(8); tiff_data.push_back(0);   // IFD offset (8)
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // IFD with 12 entries (all standard tags we support for tiled TIFF)
  tiff_data.push_back(12); tiff_data.push_back(0);  // 12 entries
  
  // Helper to add an IFD entry
  auto AddEntry = [&tiff_data](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    tiff_data.push_back(tag & 0xFF);
    tiff_data.push_back((tag >> 8) & 0xFF);
    tiff_data.push_back(type & 0xFF);
    tiff_data.push_back((type >> 8) & 0xFF);
    tiff_data.push_back(count & 0xFF);
    tiff_data.push_back((count >> 8) & 0xFF);
    tiff_data.push_back((count >> 16) & 0xFF);
    tiff_data.push_back((count >> 24) & 0xFF);
    tiff_data.push_back(value & 0xFF);
    tiff_data.push_back((value >> 8) & 0xFF);
    tiff_data.push_back((value >> 16) & 0xFF);
    tiff_data.push_back((value >> 24) & 0xFF);
  };
  
  // Add all standard tags with their test values for a tiled TIFF
  AddEntry(256, 3, 1, 2048);         // ImageWidth = 2048
  AddEntry(257, 3, 1, 2048);         // ImageLength = 2048
  AddEntry(258, 3, 1, 32);           // BitsPerSample = 32
  AddEntry(259, 3, 1, 8);            // Compression = 8 (Deflate)
  AddEntry(262, 3, 1, 1);            // PhotometricInterpretation = 1 (BlackIsZero)
  AddEntry(277, 3, 1, 1);            // SamplesPerPixel = 1
  AddEntry(284, 3, 1, 1);            // PlanarConfiguration = 1 (chunky)
  AddEntry(339, 3, 1, 3);            // SampleFormat = 3 (IEEE float)
  
  // Tile-specific tags
  AddEntry(322, 3, 1, 256);          // TileWidth = 256
  AddEntry(323, 3, 1, 256);          // TileLength = 256
  AddEntry(324, 4, 1, 1000);         // TileOffsets = 1000
  AddEntry(325, 4, 1, 10000);        // TileByteCounts = 10000
  
  // No more IFDs
  tiff_data.push_back(0); tiff_data.push_back(0);
  tiff_data.push_back(0); tiff_data.push_back(0);
  
  // Pad the file to ensure it's large enough
  while (tiff_data.size() < 2048) {
    tiff_data.push_back('X');
  }

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "tiled_tiff_all_tags.tiff", absl::Cord(tiff_data))
          .result(),
      ::tensorstore::IsOk());

  auto cache = GetCache<TiffDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<TiffDirectoryCache>(memory.driver, InlineExecutor{});
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