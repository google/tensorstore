// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/zip/zip_dir_cache.h"

#include <stddef.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Context;
using ::tensorstore::InlineExecutor;
using ::tensorstore::IsOk;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::GetCache;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_zip_kvstore::ZipDirectoryCache;
using ::tensorstore::kvstore::DriverPtr;

ABSL_FLAG(std::string, tensorstore_test_data, "",
          "Path to internal/compression/testdata/data.zip");

namespace {

absl::Cord GetTestZipFileData() {
  ABSL_CHECK(!absl::GetFlag(FLAGS_tensorstore_test_data).empty());
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(absl::GetFlag(FLAGS_tensorstore_test_data)), filedata));
  ABSL_CHECK_EQ(filedata.size(), 319482);
  return filedata;
}

std::string GetTestZipPath(std::string_view filename) {
  std::string base_path = absl::GetFlag(FLAGS_tensorstore_test_data);
  size_t last_slash = base_path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return base_path.substr(0, last_slash + 1) + std::string(filename);
  }
  return std::string(filename);
}

absl::Cord GetZipTest2Data() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_test2.zip")), filedata));
  return filedata;
}

TEST(ZipDirectoryKvsTest, Basic) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Prepare the kvstore zip file.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "data.zip", GetTestZipFileData())
          .result(),
      IsOk());

  auto cache = GetCache<ZipDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<ZipDirectoryCache>(memory.driver, InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "data.zip");
  auto status = entry->Read({absl::InfinitePast()}).status();
  ASSERT_THAT(status, IsOk());

  ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*entry);
  auto* dir = lock.data();
  ASSERT_THAT(dir, ::testing::NotNull());

  ASSERT_THAT(dir->entries, ::testing::SizeIs(3));

  EXPECT_THAT(dir->entries[0].filename, "data/a.png");
  EXPECT_THAT(dir->entries[1].filename, "data/bb.png");
  EXPECT_THAT(dir->entries[2].filename, "data/c.png");
}

TEST(ZipDirectoryKvsTest, MissingEntry) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Prepare the kvstore zip file.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  auto cache = GetCache<ZipDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<ZipDirectoryCache>(memory.driver, InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "data.zip");
  TENSORSTORE_ASSERT_OK(entry->Read({absl::InfinitePast()}));

  ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*entry);
  EXPECT_FALSE(lock.data());
}

TEST(ZipDirectoryKvsTest, MinimalZip) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  // Prepare the kvstore zip file.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(tensorstore::kvstore::Write(memory, "data.zip", GetZipTest2Data())
                  .result(),
              IsOk());

  auto cache = GetCache<ZipDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<ZipDirectoryCache>(memory.driver, InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "data.zip");
  auto status = entry->Read({absl::InfinitePast()}).status();
  ASSERT_THAT(status, IsOk());

  ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*entry);
  auto* dir = lock.data();
  ASSERT_THAT(dir, ::testing::NotNull());

  ASSERT_THAT(dir->entries, ::testing::SizeIs(2));

  EXPECT_THAT(dir->entries[0].filename, "test");
  EXPECT_THAT(dir->entries[1].filename, "testdir/test2");
}

TEST(ZipDirectoryKvsTest, RetryOnConcurrentModification) {
  auto context = Context::Default();
  auto pool = CachePool::Make(CachePool::Limits{});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      tensorstore::KvStore memory,
      tensorstore::kvstore::Open({{"driver", "memory"}}, context).result());

  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "data.zip", GetTestZipFileData())
          .result(),
      IsOk());

  auto mock_store = MockKeyValueStore::Make();

  auto cache = GetCache<ZipDirectoryCache>(pool.get(), "", [&] {
    return std::make_unique<ZipDirectoryCache>(mock_store, InlineExecutor{});
  });

  auto entry = GetCacheEntry(cache, "data.zip");
  auto future = entry->Read({absl::InfinitePast()});

  // Step 1: Pop the EOCD read request and forward to memory.
  ASSERT_EQ(1, mock_store->read_requests.size());
  mock_store->read_requests.pop()(memory.driver);

  // Step 2: Simulate concurrent modification.
  // We rewrite to "data.zip" to change its generation on memory.
  ASSERT_THAT(
      tensorstore::kvstore::Write(memory, "data.zip", GetTestZipFileData())
          .result(),
      IsOk());

  // Step 3: Pop Central Directory read request, and forward it to memory.
  // Since generation has changed, this read request fails with aborted().
  // And with the fix, this will trigger a retry from StartEOCDBlockRead().
  ASSERT_EQ(1, mock_store->read_requests.size());
  mock_store->read_requests.pop()(memory.driver);

  // Step 4: Now a new EOCD read request should be queued due to retry.
  ASSERT_EQ(1, mock_store->read_requests.size());
  mock_store->read_requests.pop()(memory.driver);

  // Step 5: Pop the new Central Directory read request.
  ASSERT_EQ(1, mock_store->read_requests.size());
  mock_store->read_requests.pop()(memory.driver);

  // Step 6: Verify the read future succeeds.
  ASSERT_THAT(future.status(), IsOk());

  ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(*entry);
  auto* dir = lock.data();
  ASSERT_THAT(dir, ::testing::NotNull());
  ASSERT_THAT(dir->entries, ::testing::SizeIs(3));
}

}  // namespace
