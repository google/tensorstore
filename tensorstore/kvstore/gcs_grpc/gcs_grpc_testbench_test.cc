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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/macros.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/client_context.h"  // third_party
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/gcs/gcs_testbench.h"
#include "tensorstore/kvstore/gcs_grpc/test_hook.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/future.h"

// Connect to an already-running testbench server on the grpc port by
// setting e.g. --testbench_grpc_endpoint=localhost:43211
ABSL_FLAG(std::string, testbench_grpc_endpoint, "", "testbench endpoint");

ABSL_FLAG(double, error_injection_probability, 0.1,
          "Probability used to inject errors inth grpc storage requests.");

namespace kvstore = ::tensorstore::kvstore;

using ::gcs_testbench::StorageTestbench;
using ::tensorstore::KvStore;
using ::tensorstore::internal::KeyValueStoreOpsTestParameters;

namespace {

StorageTestbench& GetTestBench();

struct Instructions {
  absl::Mutex mutex;
  absl::BitGen rng ABSL_GUARDED_BY(mutex);
  std::vector<std::string> test_ids ABSL_GUARDED_BY(mutex);

  size_t CreateInstructions(absl::BitGen& rng, StorageTestbench& testbench,
                            std::string_view format) ABSL_LOCKS_EXCLUDED(mutex);

  void ApplyInstructions(grpc::ClientContext* context)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    if (test_ids.empty()) {
      return;
    }
    context->AddMetadata("x-retry-test-id", test_ids.back());
    test_ids.pop_back();
  }
};

absl::NoDestructor<Instructions> get_instructions;
absl::NoDestructor<Instructions> insert_instructions;

// https://github.com/googleapis/storage-testbench?tab=readme-ov-file#retry-test-api
// These storage-testbench x-retry-test-id instructions should be
// retriable:
static constexpr std::string_view kRetryTestInstructions[] = {
    "return-429",
    "return-503",
    "return-429-after-1K",
    "return-503-after-1K",
    "return-broken-stream",
    "return-broken-stream-after-1K",
    "return-reset-connection",
};

size_t Instructions::CreateInstructions(absl::BitGen& rng,
                                        StorageTestbench& testbench,
                                        std::string_view format) {
  absl::MutexLock lock(mutex);
  auto reply = kRetryTestInstructions[absl::Uniform<size_t>(
      rng, 0, ABSL_ARRAYSIZE(kRetryTestInstructions))];

  std::string instructions = absl::Substitute(format, reply);
  auto id = testbench.CreateRetryTest(instructions);
  if (id.empty()) {
    return std::numeric_limits<size_t>::max();
  }
  test_ids.push_back(std::move(id));
  return test_ids.size();
}

const char* kGetTemplate =
    R"({"instructions":{"storage.objects.get": ["$0"]}, "transport": "GRPC"})";

const char* kInsertTemplate =
    R"({"instructions":{"storage.objects.insert": ["$0"]}, "transport": "GRPC"})";

struct ReadTestHook {
  double error_injection_probability =
      absl::GetFlag(FLAGS_error_injection_probability);

  void operator()(grpc::ClientContext* context,
                  google::storage::v2::ReadObjectRequest& request) {
    absl::MutexLock lock(get_instructions->mutex);
    auto& rng = get_instructions->rng;
    if (!absl::Bernoulli(rng, error_injection_probability)) {
      return;
    }
    // Some percentage of the time, return corrupted data.
    if (absl::Bernoulli(rng, 0.2)) {
      context->AddMetadata("x-goog-emulator-instructions",
                           "return-corrupted-data");
      if (absl::Bernoulli(rng, 0.6)) {
        return;
      }
    }
    get_instructions->ApplyInstructions(context);
  }
};

struct WriteTestHook {
  double error_injection_probability =
      absl::GetFlag(FLAGS_error_injection_probability);

  void operator()(grpc::ClientContext* context,
                  google::storage::v2::WriteObjectRequest& request) {
    absl::MutexLock lock(insert_instructions->mutex);
    auto& rng = insert_instructions->rng;
    if (!absl::Bernoulli(rng, error_injection_probability)) {
      return;
    }
    insert_instructions->ApplyInstructions(context);
  }
};

StorageTestbench& GetTestBench() {
  static absl::NoDestructor<StorageTestbench> testbench;
  [[maybe_unused]] static bool ready = [&] {
    testbench->SpawnProcess();
    auto grpc_endpoint = testbench->grpc_address();
    ABSL_LOG(INFO) << "Using " << grpc_endpoint;
    ABSL_LOG(INFO) << "Creating bucket: "
                   << StorageTestbench::CreateBucket(grpc_endpoint,
                                                     "test_bucket");

    // Only install the hooks when spawning our own testbench.
    if (absl::GetFlag(FLAGS_error_injection_probability) > 0 &&
        absl::GetFlag(FLAGS_error_injection_probability) < 1.0) {
      ABSL_LOG(INFO) << "Installing test hooks with p="
                     << absl::GetFlag(FLAGS_error_injection_probability);
      absl::BitGen rng;
      while (get_instructions->CreateInstructions(rng, *testbench,
                                                  kGetTemplate) < 100) {
        // keep going
      }
      while (insert_instructions->CreateInstructions(rng, *testbench,
                                                     kInsertTemplate) < 100) {
        // keep going
      }
      tensorstore::internal_gcs_grpc::SetTestHook(ReadTestHook{});
      tensorstore::internal_gcs_grpc::SetTestHook(WriteTestHook{});
    }
    return true;
  }();

  return *testbench;
}

std::string GetTestBenchEndpoint() {
  static std::string endpoint = [&]() -> std::string {
    std::string grpc_endpoint = absl::GetFlag(FLAGS_testbench_grpc_endpoint);
    if (!grpc_endpoint.empty()) {
      return grpc_endpoint;
    }
    return GetTestBench().grpc_address();
  }();

  return endpoint;
}

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

TENSORSTORE_GLOBAL_INITIALIZER {
  KeyValueStoreOpsTestParameters params;
  params.test_name = "Basic";
  params.get_store = [](auto callback) { callback(OpenStore()); };
  params.test_list_without_prefix = false;
  params.test_list_prefix = "list/";
  params.value_size = 2 * 1024;
  RegisterKeyValueStoreOpsTests(params);
}

TEST(GcsGrpcTestbenchTest, CancellationDoesNotCrash) {
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

TEST(GcsGrpcTestbenchTest, ConcurrentWrites) {
  tensorstore::internal::TestConcurrentWritesOptions options;
  auto store = OpenStore("concurrent_writes/");
  options.get_store = [&] { return store; };
  options.num_iterations = 0x7f;
  tensorstore::internal::TestConcurrentWrites(options);
}

TEST(GcsGrpcTestbenchTest, BatchRead) {
  // Note: With server-side retries the metrics may not match. :/
  auto store = OpenStore("batch_read/");
  tensorstore::internal::BatchReadGenericCoalescingTestOptions options;
  options.coalescing_options = tensorstore::internal_kvstore_batch::
      kDefaultRemoteStorageCoalescingOptions;

  // Don't test `target_coalesced_size` because writing a large file is too
  // slow with the fake gcs stubby implementation.
  options.coalescing_options.target_coalesced_size =
      std::numeric_limits<int64_t>::max();

  options.metric_prefix = "/tensorstore/kvstore/gcs_grpc/";
  tensorstore::internal::TestBatchReadGenericCoalescing(store, options);
}

}  // namespace
