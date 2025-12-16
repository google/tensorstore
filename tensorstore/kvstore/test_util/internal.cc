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

#include "tensorstore/kvstore/test_util/internal.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/kvstore/driver.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

StorageGeneration GetStorageGeneration(const KvStore& store, std::string key) {
  auto get = kvstore::Read(store, key).result();
  StorageGeneration gen;
  if (get.ok()) {
    gen = get->stamp.generation;
  }
  return gen;
}

// Return a highly-improbable storage generation
StorageGeneration GetMismatchStorageGeneration(const KvStore& store) {
  auto spec_result = store.spec();

  if (spec_result.ok() && spec_result->driver->driver_id() == "s3") {
    return StorageGeneration::FromString("\"abcdef1234567890\"");
  }

  // Use a single uint64_t storage generation here for GCS compatibility.
  // Also, the generation looks like a nanosecond timestamp.
  return StorageGeneration::FromValues(uint64_t{/*3.*/ 1415926535897932});
}

Result<std::map<kvstore::Key, kvstore::Value>> GetMap(const KvStore& store) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto entries, ListFuture(store).result());
  std::map<kvstore::Key, kvstore::Value> result;
  for (const auto& entry : entries) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 kvstore::Read(store, entry.key).result());
    assert(!read_result.aborted());
    assert(!read_result.not_found());
    result.emplace(entry.key, std::move(read_result.value));
  }
  return result;
}

std::vector<::nlohmann::json> CollectedMetricsToJson(
    const std::vector<internal_metrics::CollectedMetric>& collected_metrics,
    bool include_zero_metrics) {
  std::vector<::nlohmann::json> lines;
  for (const auto& collected_metric : collected_metrics) {
    if (include_zero_metrics ||
        internal_metrics::IsCollectedMetricNonZero(collected_metric)) {
      lines.push_back(
          internal_metrics::CollectedMetricToJson(collected_metric));
    }
  }
  return lines;
}

}  // namespace internal
}  // namespace tensorstore