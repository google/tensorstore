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

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(6)
                       .AddEntry(256, 3, 1, 256)
                       .AddEntry(257, 3, 1, 256)
                       .AddEntry(322, 3, 1, 256)
                       .AddEntry(323, 3, 1, 256)
                       .AddEntry(324, 4, 1, 128)
                       .AddEntry(325, 4, 1, 256)
                       .EndIfd()
                       .PadTo(2048)
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

    EXPECT_EQ(data->directories.size(), 1);
    EXPECT_EQ(data->directories[0].entries.size(), 6);
    EXPECT_EQ(data->image_directories.size(), 1);

    EXPECT_EQ(data->image_directories[0].width, 256);
    EXPECT_EQ(data->image_directories[0].height, 256);
    EXPECT_EQ(data->image_directories[0].is_tiled, true);
    EXPECT_EQ(data->image_directories[0].chunk_width, 256);
    EXPECT_EQ(data->image_directories[0].chunk_height, 256);
  }
}

TEST(TiffDirectoryCacheTest, ReadFull) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

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

  {
    tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
    request.staleness_bound = absl::InfinitePast();

    ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
    auto* data = lock.data();
    ASSERT_THAT(data, ::testing::NotNull());
    EXPECT_TRUE(data->full_read);

    EXPECT_EQ(data->directories.size(), 1);
    EXPECT_EQ(data->directories[0].entries.size(), 5);
    EXPECT_EQ(data->image_directories.size(), 1);

    EXPECT_EQ(data->image_directories[0].width, 400);
    EXPECT_EQ(data->image_directories[0].height, 300);
    EXPECT_EQ(data->image_directories[0].is_tiled, false);
    EXPECT_EQ(data->image_directories[0].chunk_height, 100);
    EXPECT_EQ(data->image_directories[0].chunk_offsets.size(), 1);
    EXPECT_EQ(data->image_directories[0].chunk_offsets[0], 128);
    EXPECT_EQ(data->image_directories[0].chunk_bytecounts.size(), 1);
    EXPECT_EQ(data->image_directories[0].chunk_bytecounts[0], 200);
  }
}

TEST(TiffDirectoryCacheTest, BadIfdFailsParse) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  TiffBuilder builder;
  // Claim 10 entries (too many)
  auto corrupt_tiff = builder.StartIfd(10).AddEntry(1, 1, 1, 0).Build();

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

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  uint32_t strip_offsets_offset = 200;
  uint32_t strip_bytecounts_offset = 216;
  uint32_t strip_offsets[4] = {1000, 2000, 3000, 4000};
  uint32_t strip_bytecounts[4] = {500, 600, 700, 800};

  TiffBuilder builder;
  auto tiff_data =
      builder.StartIfd(5)
          .AddEntry(256, 3, 1, 800)
          .AddEntry(257, 3, 1, 600)
          .AddEntry(278, 3, 1, 100)
          .AddEntry(273, 4, 4, strip_offsets_offset)
          .AddEntry(279, 4, 4, strip_bytecounts_offset)
          .EndIfd()
          .PadTo(strip_offsets_offset)
          .AddUint32Array({strip_offsets[0], strip_offsets[1], strip_offsets[2],
                           strip_offsets[3]})
          .AddUint32Array({strip_bytecounts[0], strip_bytecounts[1],
                           strip_bytecounts[2], strip_bytecounts[3]})
          .PadTo(4096)
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

  {
    tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
    request.staleness_bound = absl::InfinitePast();

    ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
    auto* data = lock.data();
    ASSERT_THAT(data, ::testing::NotNull());

    EXPECT_EQ(data->image_directories[0].chunk_offsets.size(), 4);
    EXPECT_EQ(data->image_directories[0].chunk_bytecounts.size(), 4);

    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(data->image_directories[0].chunk_offsets[i], strip_offsets[i]);
      EXPECT_EQ(data->image_directories[0].chunk_bytecounts[i],
                strip_bytecounts[i]);
    }
  }
}

TEST(TiffDirectoryCacheTest, ExternalArrays_BadPointer) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  uint32_t invalid_offset = 50000;  // Far beyond our file size

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(5)
                       .AddEntry(256, 3, 1, 800)
                       .AddEntry(257, 3, 1, 600)
                       .AddEntry(278, 3, 1, 100)
                       .AddEntry(273, 4, 4, invalid_offset)
                       .AddEntry(279, 4, 1, 500)
                       .EndIfd()
                       .PadTo(1000)
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

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  auto read_result = entry->Read(request).result();
  EXPECT_THAT(read_result.status(), ::testing::Not(::tensorstore::IsOk()));

  EXPECT_TRUE(absl::IsOutOfRange(read_result.status()) ||
              absl::IsDataLoss(read_result.status()) ||
              absl::IsInvalidArgument(read_result.status()) ||
              absl::IsFailedPrecondition(read_result.status()));
}

// Helper to create a test TIFF file with multiple IFDs
std::string MakeMultiPageTiff() {
  TiffBuilder builder;

  return builder.StartIfd(5)
      .AddEntry(256, 3, 1, 400)
      .AddEntry(257, 3, 1, 100)
      .AddEntry(278, 3, 1, 100)
      .AddEntry(273, 4, 1, 1000)
      .AddEntry(279, 4, 1, 200)
      .EndIfd(200)
      .PadTo(200)
      .StartIfd(6)
      .AddEntry(256, 3, 1, 256)
      .AddEntry(257, 3, 1, 256)
      .AddEntry(322, 3, 1, 256)
      .AddEntry(323, 3, 1, 256)
      .AddEntry(324, 4, 1, 2000)
      .AddEntry(325, 4, 1, 300)
      .EndIfd()
      .PadTo(3000)
      .Build();
}

TEST(TiffDirectoryCacheMultiIfdTest, ReadAndVerifyIFDs) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

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

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  // Check first IFD (strip-based)
  const auto& ifd1 = data->directories[0];
  const auto& img1 = data->image_directories[0];
  EXPECT_EQ(ifd1.entries.size(), 5);
  EXPECT_EQ(img1.width, 400);
  EXPECT_EQ(img1.height, 100);
  EXPECT_EQ(img1.is_tiled, false);
  EXPECT_EQ(img1.chunk_height, 100);
  EXPECT_EQ(img1.chunk_offsets.size(), 1);
  EXPECT_EQ(img1.chunk_offsets[0], 1000);
  EXPECT_EQ(img1.chunk_bytecounts[0], 200);

  // Check second IFD (tile-based)
  const auto& ifd2 = data->directories[1];
  const auto& img2 = data->image_directories[1];
  EXPECT_EQ(ifd2.entries.size(), 6);
  EXPECT_EQ(img2.width, 256);
  EXPECT_EQ(img2.height, 256);
  EXPECT_EQ(img2.is_tiled, true);
  EXPECT_EQ(img2.chunk_width, 256);
  EXPECT_EQ(img2.chunk_height, 256);
  EXPECT_EQ(img2.chunk_offsets.size(), 1);
  EXPECT_EQ(img2.chunk_offsets[0], 2000);

  EXPECT_FALSE(data->full_read);
}

TEST(TiffDirectoryCacheMultiIfdTest, ReadLargeMultiPageTiff) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  // Create a TIFF file larger than kInitialReadBytes
  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(5)
                       .AddEntry(256, 3, 1, 400)
                       .AddEntry(257, 3, 1, 300)
                       .AddEntry(278, 3, 1, 100)
                       .AddEntry(273, 4, 1, 1024)
                       .AddEntry(279, 4, 1, 200)
                       .EndIfd(2048)
                       .PadTo(2048)
                       .StartIfd(6)
                       .AddEntry(256, 3, 1, 256)
                       .AddEntry(257, 3, 1, 256)
                       .AddEntry(322, 3, 1, 256)
                       .AddEntry(323, 3, 1, 256)
                       .AddEntry(324, 4, 1, 3000)
                       .AddEntry(325, 4, 1, 300)
                       .EndIfd()
                       .PadTo(4096)
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

  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  EXPECT_EQ(data->image_directories[0].width, 400);
  EXPECT_EQ(data->image_directories[1].width, 256);
}

TEST(TiffDirectoryCacheMultiIfdTest, ExternalArraysMultiIfdTest) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  std::vector<uint32_t> offsets1 = {1000, 2000, 3000, 4000};
  std::vector<uint32_t> bytecounts1 = {50, 60, 70, 80};
  std::vector<uint32_t> offsets2 = {5000, 5004, 5008, 5012};
  std::vector<uint32_t> bytecounts2 = {100, 200, 300, 400};

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(5)
                       .AddEntry(256, 3, 1, 400)
                       .AddEntry(257, 3, 1, 300)
                       .AddEntry(278, 3, 1, 100)
                       .AddEntry(273, 4, 4, 512)
                       .AddEntry(279, 4, 4, 528)
                       .EndIfd(600)
                       .PadTo(512)
                       .AddUint32Array(offsets1)
                       .AddUint32Array(bytecounts1)
                       .PadTo(600)
                       .StartIfd(6)
                       .AddEntry(256, 3, 1, 512)
                       .AddEntry(257, 3, 1, 512)
                       .AddEntry(322, 3, 1, 256)
                       .AddEntry(323, 3, 1, 256)
                       .AddEntry(324, 4, 4, 700)
                       .AddEntry(325, 4, 4, 716)
                       .EndIfd()
                       .PadTo(700)
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

  EXPECT_EQ(data->directories.size(), 2);
  EXPECT_EQ(data->image_directories.size(), 2);

  EXPECT_EQ(data->image_directories[0].chunk_offsets.size(), 4);
  EXPECT_EQ(data->image_directories[0].chunk_bytecounts.size(), 4);

  EXPECT_EQ(data->image_directories[1].chunk_offsets.size(), 4);
  EXPECT_EQ(data->image_directories[1].chunk_bytecounts.size(), 4);
}

TEST(TiffDirectoryCacheTest, ExternalArrays_Uint16Arrays) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  uint32_t bits_per_sample_offset = 200;
  uint32_t sample_format_offset = 212;
  std::vector<uint16_t> bits_values = {8, 8, 8};
  std::vector<uint16_t> sample_format_values = {1, 1, 1};

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(8)
                       .AddEntry(256, 3, 1, 800)
                       .AddEntry(257, 3, 1, 600)
                       .AddEntry(277, 3, 1, 3)
                       .AddEntry(278, 3, 1, 100)
                       .AddEntry(258, 3, 3, bits_per_sample_offset)
                       .AddEntry(339, 3, 3, sample_format_offset)
                       .AddEntry(273, 4, 1, 1000)
                       .AddEntry(279, 4, 1, 30000)
                       .EndIfd()
                       .PadTo(bits_per_sample_offset)
                       .AddUint16Array(bits_values)
                       .PadTo(sample_format_offset)
                       .AddUint16Array(sample_format_values)
                       .PadTo(2048)
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

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  const auto& img_dir = data->image_directories[0];

  EXPECT_EQ(img_dir.samples_per_pixel, 3);
  EXPECT_EQ(img_dir.chunk_height, 100);
  ASSERT_EQ(img_dir.bits_per_sample.size(), 3);

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(img_dir.bits_per_sample[i], bits_values[i]);
  }

  ASSERT_EQ(img_dir.sample_format.size(), 3);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(img_dir.sample_format[i], sample_format_values[i]);
  }
}

// Comprehensive test that checks all supported TIFF tags
TEST(TiffDirectoryCacheTest, ComprehensiveTiffTagsTest) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(11)
                       .AddEntry(256, 3, 1, 1024)
                       .AddEntry(257, 3, 1, 768)
                       .AddEntry(258, 3, 1, 16)
                       .AddEntry(259, 3, 1, 1)
                       .AddEntry(262, 3, 1, 2)
                       .AddEntry(277, 3, 1, 1)
                       .AddEntry(278, 3, 1, 128)
                       .AddEntry(273, 4, 1, 1000)
                       .AddEntry(279, 4, 1, 65536)
                       .AddEntry(284, 3, 1, 1)
                       .AddEntry(339, 3, 1, 1)
                       .EndIfd()
                       .PadTo(2048)
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

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  const auto& img_dir = data->image_directories[0];
  EXPECT_EQ(img_dir.width, 1024);
  EXPECT_EQ(img_dir.height, 768);
  ASSERT_EQ(img_dir.bits_per_sample.size(), 1);
  EXPECT_EQ(img_dir.bits_per_sample[0], 16);
  EXPECT_EQ(img_dir.compression, 1);
  EXPECT_EQ(img_dir.photometric, 2);
  EXPECT_EQ(img_dir.samples_per_pixel, 1);
  EXPECT_EQ(img_dir.is_tiled, false);
  EXPECT_EQ(img_dir.chunk_height, 128);
  ASSERT_EQ(img_dir.chunk_offsets.size(), 1);
  EXPECT_EQ(img_dir.chunk_offsets[0], 1000);
  ASSERT_EQ(img_dir.chunk_bytecounts.size(), 1);
  EXPECT_EQ(img_dir.chunk_bytecounts[0], 65536);
  EXPECT_EQ(img_dir.planar_config, 1);
  ASSERT_EQ(img_dir.sample_format.size(), 1);
  EXPECT_EQ(img_dir.sample_format[0], 1);
}

// Test for a tiled TIFF with all supported tags
TEST(TiffDirectoryCacheTest, TiledTiffWithAllTags) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  TiffBuilder builder;
  auto tiff_data = builder.StartIfd(12)
                       .AddEntry(256, 3, 1, 256)
                       .AddEntry(257, 3, 1, 256)
                       .AddEntry(258, 3, 1, 32)
                       .AddEntry(259, 3, 1, 8)
                       .AddEntry(262, 3, 1, 1)
                       .AddEntry(277, 3, 1, 1)
                       .AddEntry(284, 3, 1, 1)
                       .AddEntry(339, 3, 1, 3)
                       .AddEntry(322, 3, 1, 256)
                       .AddEntry(323, 3, 1, 256)
                       .AddEntry(324, 4, 1, 1000)
                       .AddEntry(325, 4, 1, 10000)
                       .EndIfd()
                       .PadTo(2048)
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

  tensorstore::internal::AsyncCache::AsyncCacheReadRequest request;
  request.staleness_bound = absl::InfinitePast();

  ASSERT_THAT(entry->Read(request).result(), ::tensorstore::IsOk());

  TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(*entry);
  auto* data = lock.data();
  ASSERT_THAT(data, ::testing::NotNull());

  const auto& img_dir = data->image_directories[0];

  EXPECT_EQ(img_dir.width, 256);
  EXPECT_EQ(img_dir.height, 256);
  ASSERT_EQ(img_dir.bits_per_sample.size(), 1);
  EXPECT_EQ(img_dir.bits_per_sample[0], 32);
  EXPECT_EQ(img_dir.compression, 8);
  EXPECT_EQ(img_dir.photometric, 1);
  EXPECT_EQ(img_dir.samples_per_pixel, 1);
  EXPECT_EQ(img_dir.planar_config, 1);
  ASSERT_EQ(img_dir.sample_format.size(), 1);
  EXPECT_EQ(img_dir.sample_format[0], 3);

  EXPECT_EQ(img_dir.chunk_width, 256);
  EXPECT_EQ(img_dir.chunk_height, 256);
  ASSERT_EQ(img_dir.chunk_offsets.size(), 1);
  EXPECT_EQ(img_dir.chunk_offsets[0], 1000);
  ASSERT_EQ(img_dir.chunk_bytecounts.size(), 1);
  EXPECT_EQ(img_dir.chunk_bytecounts[0], 10000);
}

}  // namespace