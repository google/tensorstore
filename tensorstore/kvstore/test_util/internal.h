// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_INTERNAL_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

class Cleanup {
 public:
  Cleanup(KvStore store, std::vector<std::string> keys)
      : store_(std::move(store)), keys_(std::move(keys)) {
    DoCleanup();
  }

  void DoCleanup() {
    // Delete everything that we're going to use before starting.
    // This is helpful if, for instance, we run against a persistent
    // service and the test crashed half-way through last time.
    ABSL_LOG(INFO) << "Cleanup";
    for (const auto& to_remove : keys_) {
      TENSORSTORE_CHECK_OK(kvstore::Delete(store_, to_remove).result());
    }
  }

  ~Cleanup() { DoCleanup(); }

 private:
  KvStore store_;
  std::vector<std::string> keys_;
};

/// Returns the current time as of the start of the call, and waits until that
/// time is no longer the current time.
///
/// This is used to ensure consistent testing.
inline absl::Time UniqueNow(absl::Duration epsilon = absl::Nanoseconds(1)) {
  absl::Time t = absl::Now();
  do {
    absl::SleepFor(absl::Milliseconds(1));
  } while (absl::Now() < t + epsilon);
  return t;
}

// Returns the storage generation of `key` in the `kvstore`.
StorageGeneration GetStorageGeneration(const KvStore& store, std::string key);

// Return a highly-improbable storage generation
StorageGeneration GetMismatchStorageGeneration(const KvStore& store);

// Returns the contents of `kv_store` as an `std::map`.
Result<std::map<kvstore::Key, kvstore::Value>> GetMap(const KvStore& store);

// Converts a vector of CollectedMetrics to a vector of json objects, filtering
// out zero metrics if `include_zero_metrics` is false.
std::vector<::nlohmann::json> CollectedMetricsToJson(
    const std::vector<internal_metrics::CollectedMetric>& collected_metrics,
    bool include_zero_metrics = false);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_INTERNAL_H_
