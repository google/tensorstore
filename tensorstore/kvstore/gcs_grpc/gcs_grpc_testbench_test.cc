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

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/gcs/gcs_testbench.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

// Connect to an already-running testbench server on the grpc port.
ABSL_FLAG(std::string, testbench_grpc_endpoint, "", "testbench endpoint");

namespace kvstore = ::tensorstore::kvstore;

using ::gcs_testbench::StorageTestbench;
using ::tensorstore::KvStore;
using ::tensorstore::StorageGeneration;

namespace {

std::string GetTestBenchEndpoint() {
  static absl::NoDestructor<StorageTestbench> testbench;
  static std::string endpoint = [&] {
    std::string grpc_endpoint = absl::GetFlag(FLAGS_testbench_grpc_endpoint);
    if (grpc_endpoint.empty()) {
      testbench->SpawnProcess();
      grpc_endpoint = testbench->grpc_address();
    }
    ABSL_LOG(INFO) << "Using " << grpc_endpoint;
    ABSL_LOG(INFO) << "Creating bucket: "
                   << StorageTestbench::CreateBucket(grpc_endpoint,
                                                     "test_bucket");
    return grpc_endpoint;
  }();

  return endpoint;
}

class GcsGrpcTestbenchTest : public testing::Test {
 public:
  tensorstore::KvStore OpenStore(std::string path = "") {
    std::string testbench_grpc_endpoint = GetTestBenchEndpoint();
    return kvstore::Open({{"driver", "gcs_grpc"},
                          {"endpoint", testbench_grpc_endpoint},
                          {"timeout", "200ms"},
                          {"num_channels", 1},
                          {"bucket", "test_bucket"},
                          {"path", path}})
        .value();
  }
};

TEST_F(GcsGrpcTestbenchTest, Basic) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueReadWriteOps(store);
}

TEST_F(GcsGrpcTestbenchTest, DeletePrefix) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeletePrefix(store);
}

TEST_F(GcsGrpcTestbenchTest, DeleteRange) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRange(store);
}

TEST_F(GcsGrpcTestbenchTest, DeleteRangeToEnd) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeToEnd(store);
}

TEST_F(GcsGrpcTestbenchTest, DeleteRangeFromBeginning) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreDeleteRangeFromBeginning(store);
}

TEST_F(GcsGrpcTestbenchTest, List) {
  auto store = OpenStore("list/");
  tensorstore::internal::TestKeyValueStoreList(store);
}

TEST_F(GcsGrpcTestbenchTest, CancellationDoesNotCrash) {
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

TEST_F(GcsGrpcTestbenchTest, ConcurrentWrites) {
  tensorstore::internal::TestConcurrentWritesOptions options;
  auto store = OpenStore("concurrent_writes/");
  options.get_store = [&] { return store; };
  options.num_iterations = 0x7f;
  tensorstore::internal::TestConcurrentWrites(options);
}

TEST_F(GcsGrpcTestbenchTest, BatchRead) {
  auto store = OpenStore("batch_read/");
  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options = tensorstore::internal_kvstore_batch::
      kDefaultRemoteStorageCoalescingOptions;

  // Don't test `target_coalesced_size` because writing a large file is too slow
  // with the fake gcs stubby implementation.
  options.coalescing_options.target_coalesced_size =
      std::numeric_limits<int64_t>::max();

  options.metric_prefix = "/tensorstore/kvstore/gcs_grpc/";
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}

}  // namespace
