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

#include "tensorstore/internal/cache/kvs_backed_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string_view>

#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/domain_field.h"  // iwyu: keep
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registration.h"

namespace tensorstore {
namespace internal {
namespace {
struct KvsCacheReadDomain {
  static constexpr std::array<std::string_view, 3> kValues = {
      "unchanged", "changed", "error"};
  // FIND_SEED unchanged changed error
  static constexpr uint32_t kSeed = 2;
  static constexpr size_t kTableSize = 3;
};

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    kvs_cache_read, (Counter<int64_t, DomainField<KvsCacheReadDomain>>),
    MetricMetadata(
        "/tensorstore/cache/kvs_cache_read",
        "Count of kvs_backed_cache reads by category. A large number of "
        "'unchanged' reads indicates that the dataset is relatively "
        "quiescent."),
    "category");

}  // namespace

void KvsBackedCache_IncrementReadUnchangedMetric() {
  static auto& cell = kvs_cache_read.GetCell("unchanged");
  cell.Increment();
}

void KvsBackedCache_IncrementReadChangedMetric() {
  static auto& cell = kvs_cache_read.GetCell("changed");
  cell.Increment();
}

void KvsBackedCache_IncrementReadErrorMetric() {
  static auto& cell = kvs_cache_read.GetCell("error");
  cell.Increment();
}

}  // namespace internal
}  // namespace tensorstore
