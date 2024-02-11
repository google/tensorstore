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

#include "tensorstore/kvstore/gcs/gcs_testbench.h"

#include <stddef.h>

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace kvstore = ::tensorstore::kvstore;

using ::gcs_testbench::StorageTestbench;
using ::tensorstore::KvStore;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::NoDestructor;

namespace {

// TODO: Move this to a public location.

StorageTestbench& GetTestBench() {
  static NoDestructor<StorageTestbench> testbench;

  static absl::once_flag init_once;
  absl::call_once(init_once, [&]() {
    testbench->SpawnProcess();
    static std::string http_address = testbench->http_address();
    ::tensorstore::internal::SetEnv("TENSORSTORE_GCS_HTTP_URL",
                                    http_address.c_str());
    ::tensorstore::internal::SetEnv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "abc");

    ABSL_LOG(INFO) << "Using " << http_address;
    ABSL_LOG(INFO) << "Creating bucket: "
                   << StorageTestbench::CreateBucket(testbench->grpc_address(),
                                                     "test_bucket");
  });

  return *testbench;
}

class GcsTestbenchTest : public testing::Test {
 public:
  tensorstore::KvStore OpenStore(std::string path = "") {
    GetTestBench();
    return kvstore::Open(
               {{"driver", "gcs"}, {"bucket", "test_bucket"}, {"path", path}})
        .value();
  }
};

TEST_F(GcsTestbenchTest, Basic) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(GcsTestbenchTest, DeletePrefix) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST_F(GcsTestbenchTest, DeleteRange) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST_F(GcsTestbenchTest, DeleteRangeToEnd) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST_F(GcsTestbenchTest, DeleteRangeFromBeginning) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

TEST_F(GcsTestbenchTest, List) {
  auto store = OpenStore("list/");
  tensorstore::internal::TestKeyValueStoreList(store);
}

TEST_F(GcsTestbenchTest, CancellationDoesNotCrash) {
  // There's no way to really test cancellation reasonably for Read/Write,
  // so this test issues a bunch of writes and reads, and then cancels them
  // by dropping the futures, and verifies that it does not crash.
  auto store = OpenStore("cancellation/");
  static constexpr size_t kCount = 1000;

  std::vector<std::string> keys;
  keys.reserve(kCount);

  for (size_t i = 0; i < kCount; ++i) {
    keys.push_back(absl::StrCat(i));
  }

  absl::Cord value("xyzzyx");
  std::vector<tensorstore::AnyFuture> futures;
  futures.reserve(kCount * 2);
  for (const auto& key : keys) {
    futures.push_back(kvstore::Write(store, key, value));
  }
  for (const auto& key : keys) {
    futures.push_back(kvstore::Read(store, key));
  }

  // ... And drop the futures.
  futures = {};
  for (const auto& key : keys) {
    futures.push_back(kvstore::Delete(store, key));
  }
  for (auto& future : futures) {
    future.Wait();
  }
}

struct ConcurrentWriteFn {
  static constexpr char kKey[] = "test";
  static constexpr size_t kNumIterations = 0x3f;

  const size_t offset;
  mutable std::string value;
  mutable StorageGeneration generation;
  tensorstore::KvStore store;

  void operator()() const {
    bool read = false;
    for (size_t i = 0; i < kNumIterations; /**/) {
      if (read) {
        auto read_result = kvstore::Read(store, kKey).result();
        ABSL_CHECK_OK(read_result.status());
        ABSL_CHECK(!read_result->aborted());
        ABSL_CHECK(!read_result->not_found());
        ABSL_CHECK_EQ(read_result->value.size(), value.size());
        value = std::string(read_result->value);
        generation = read_result->stamp.generation;
      }

      size_t x;
      std::memcpy(&x, &value[offset], sizeof(size_t));
      ABSL_CHECK_EQ(i, x);
      std::string new_value = value;
      x = i + 1;
      std::memcpy(&new_value[offset], &x, sizeof(size_t));
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto write_result,
          kvstore::Write(store, kKey, absl::Cord(new_value), {generation})
              .result());
      if (!StorageGeneration::IsUnknown(write_result.generation)) {
        generation = write_result.generation;
        value = new_value;
        i = x;
        read = false;
      } else {
        read = true;
      }
    }
  }
};

// On windows, this concurrent test is flaky when used against the gcs
// storage-testbench.
TEST_F(GcsTestbenchTest, ConcurrentWrites) {
  static constexpr size_t kNumThreads = 4;

  std::vector<tensorstore::internal::Thread> threads;
  threads.reserve(kNumThreads);

  auto store = OpenStore("concurrent_writes/");
  std::string initial_value;
  initial_value.resize(sizeof(size_t) * kNumThreads);
  StorageGeneration initial_generation =
      kvstore::Write(store, ConcurrentWriteFn::kKey, absl::Cord(initial_value))
          .value()
          .generation;

  for (size_t thread_i = 0; thread_i < kNumThreads; ++thread_i) {
    threads.emplace_back(tensorstore::internal::Thread(
        {"concurrent_write"},
        ConcurrentWriteFn{thread_i * sizeof(size_t), initial_value,
                          initial_generation, store}));
  }
  for (auto& t : threads) t.Join();

  // Verify the output.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto read_result,
        kvstore::Read(store, ConcurrentWriteFn::kKey).result());
    ASSERT_FALSE(read_result.aborted() || read_result.not_found());
    auto value = std::string(read_result.value);
    for (size_t thread_i = 0; thread_i < kNumThreads; ++thread_i) {
      size_t x = 0;
      std::memcpy(&x, &value[thread_i * sizeof(size_t)], sizeof(size_t));
      EXPECT_EQ(x, ConcurrentWriteFn::kNumIterations) << thread_i;
    }
  }
}

}  // namespace
