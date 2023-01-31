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

#include "tensorstore/internal/metrics/counter.h"

namespace tensorstore {
namespace internal {
namespace {

auto& kvs_cache_read = internal_metrics::Counter<int64_t, std::string>::New(
    "/tensorstore/cache/kvs_cache_read", "category",
    "Count of kvs_backed_cache reads by category. A large number of "
    "'unchanged' reads indicates that the dataset is relatively quiescent.");
}

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
