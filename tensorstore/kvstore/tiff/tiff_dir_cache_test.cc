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
  
  // Since our test file is smaller than kInitialReadBytes (1024),
  // it should be fully read in one shot
  EXPECT_TRUE(data->full_read);
}


}  // namespace