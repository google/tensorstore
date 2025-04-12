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
    EXPECT_EQ(data->raw_data.size(), 1024);
    EXPECT_FALSE(data->full_read);
    
    // Check parsed IFD entries
    EXPECT_EQ(data->ifd_entries.size(), 6);
    
    // Check parsed image directory
    EXPECT_EQ(data->image_directory.width, 800);
    EXPECT_EQ(data->image_directory.height, 600);
    EXPECT_EQ(data->image_directory.tile_width, 256);
    EXPECT_EQ(data->image_directory.tile_height, 256);
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
    EXPECT_EQ(data->raw_data.size(), 512);
    EXPECT_TRUE(data->full_read);
    
    // Check parsed IFD entries
    EXPECT_EQ(data->ifd_entries.size(), 5);
    
    // Check parsed image directory
    EXPECT_EQ(data->image_directory.width, 400);
    EXPECT_EQ(data->image_directory.height, 300);
    EXPECT_EQ(data->image_directory.rows_per_strip, 100);
    EXPECT_EQ(data->image_directory.strip_offsets.size(), 1);
    EXPECT_EQ(data->image_directory.strip_offsets[0], 128);
    EXPECT_EQ(data->image_directory.strip_bytecounts.size(), 1);
    EXPECT_EQ(data->image_directory.strip_bytecounts[0], 200);
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

}  // namespace