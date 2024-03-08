// Copyright 2022 The TensorStore Authors
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

/// \file kvstore_duration attempts to run a kvstore with a number of parallel
/// requests over a specific duration.
///
/* Examples

bazel run -c opt \
  //tensorstore/internal/benchmark:kvstore_duration -- \
  --kvstore_spec='"file:///tmp/kvstore"' --duration=1m
*/

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/benchmark/metric_utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          {},
          "KvStore spec for reading data.  See examples at the start of the "
          "source file.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.");

ABSL_FLAG(absl::Duration, duration, absl::Seconds(20), "Duration of read loop");

ABSL_FLAG(size_t, parallelism, 100, "Read parallelism");
ABSL_FLAG(size_t, max_reads, 1000000, "Maximum read operations");

namespace tensorstore {
namespace {

auto& read_throughput = internal_metrics::Value<double>::New(
    "/tensorstore/kvstore_benchmark/read_throughput",
    "the read throughput in this test");

struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  std::vector<std::string> keys;
  tensorstore::KvStore kvstore;
  absl::Time start_time;
  absl::Time end_time;
  absl::InsecureBitGen gen;
  std::atomic<int64_t> bytes_read{0};
  std::atomic<size_t> files_read{0};
  absl::Mutex mu;

  // Record metrics from prior read.
  void RecordMetrics(Result<kvstore::ReadResult> result);

  // Start another read.
  void StartNextRead(tensorstore::Promise<void> promise);

  // Output elapsed stats.
  void OutputElapsed();
};

void ReadState::RecordMetrics(Result<kvstore::ReadResult> result) {
  files_read.fetch_add(1);
  if (!result.ok()) return;
  bytes_read.fetch_add(result->value.size());
}

void ReadState::StartNextRead(tensorstore::Promise<void> promise) {
  if (absl::Now() > end_time) return;
  if (files_read.load() >= absl::GetFlag(FLAGS_max_reads)) return;

  // Start next read. This maintains a consistent N reads in flight
  // until the test has completed.
  std::string_view key;
  {
    absl::MutexLock l(&mu);
    key = keys[absl::Uniform(gen, 0u, keys.size())];
  }

  LinkValue(
      [self = internal::IntrusivePtr<ReadState>{this}](
          tensorstore::Promise<void> promise,
          tensorstore::Future<kvstore::ReadResult> future) {
        self->RecordMetrics(future.result());
        self->StartNextRead(std::move(promise));
      },
      std::move(promise), kvstore::Read(kvstore, key));
}

void ReadState::OutputElapsed() {
  auto bytes = bytes_read.load();
  auto files = files_read.load();
  auto elapsed_s =
      absl::FDivDuration(absl::Now() - start_time, absl::Seconds(1));
  double read_mb = static_cast<double>(bytes) / 1e6;

  double throughput = read_mb / elapsed_s;
  std::cout << "Read: "
            << absl::StrFormat("%d bytes in %.0f ms:  %.3f MB/second (%d ops)",
                               bytes, elapsed_s * 1e3, throughput, files)
            << std::endl;
  read_throughput.Set(throughput);
}

void DoDurationBenchmark(Context context, kvstore::Spec kvstore_spec) {
  std::cout << "Starting read duration benchmark for "
            << absl::GetFlag(FLAGS_duration) << " with parallelism "
            << absl::GetFlag(FLAGS_parallelism) << ", reading at most "
            << absl::GetFlag(FLAGS_max_reads) << " reads" << std::endl;

  auto read_state = internal::MakeIntrusivePtr<ReadState>();

  TENSORSTORE_CHECK_OK_AND_ASSIGN(
      read_state->kvstore, kvstore::Open(kvstore_spec, context).result());

  TENSORSTORE_CHECK_OK_AND_ASSIGN(
      auto entries, kvstore::ListFuture(read_state->kvstore).result());
  ABSL_LOG(INFO) << "Read " << entries.size() << " keys from kvstore";
  ABSL_CHECK(!entries.empty());

  read_state->keys.reserve(entries.size());
  for (auto& entry : entries) {
    read_state->keys.push_back(std::move(entry.key));
  }

  auto pair = PromiseFuturePair<void>::Make(absl::OkStatus());
  read_state->start_time = absl::Now();
  read_state->end_time = read_state->start_time + absl::GetFlag(FLAGS_duration);

  for (size_t i = 0; i < absl::GetFlag(FLAGS_parallelism); i++) {
    read_state->StartNextRead(pair.promise);
  }

  // Wait until all reads are complete.
  pair.promise = {};
  pair.future.Force();
  while (!pair.future.WaitFor(absl::Seconds(10))) {
    read_state->OutputElapsed();
  }
  TENSORSTORE_CHECK_OK(pair.future.result());
  std::cout << "Done" << std::endl;
  read_state->OutputElapsed();
}

void Run() {
  ABSL_CHECK(absl::GetFlag(FLAGS_duration) > absl::ZeroDuration());
  ABSL_CHECK(absl::GetFlag(FLAGS_duration) != absl::InfiniteDuration());
  ABSL_CHECK(absl::GetFlag(FLAGS_parallelism) > 0);

  auto kvstore_spec = absl::GetFlag(FLAGS_kvstore_spec).value;
  internal::EnsureDirectoryPath(kvstore_spec.path);

  Context context(absl::GetFlag(FLAGS_context_spec).value);

  DoDurationBenchmark(context, kvstore_spec);

  internal::DumpMetrics("");
}

}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  tensorstore::Run();
  return 0;
}
