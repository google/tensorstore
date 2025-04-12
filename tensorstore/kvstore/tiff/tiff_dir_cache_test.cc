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

// Creates test data of specified size filled with 'X' pattern
absl::Cord CreateTestData(size_t size) {
  return absl::Cord(std::string(size, 'X'));
}

TEST(TiffDirectoryCacheTest, ReadSlice) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", CreateTestData(2048))
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
  }
}

TEST(TiffDirectoryCacheTest, ReadFull) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Create an in-memory kvstore with test data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "test.tiff", CreateTestData(512))
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
  }
}

}  // namespace