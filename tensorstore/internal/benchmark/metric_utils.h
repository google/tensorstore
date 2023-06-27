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

#ifndef TENSORSTORE_INTERNAL_BENCHMARK_METRIC_UTILS_H_
#define TENSORSTORE_INTERNAL_BENCHMARK_METRIC_UTILS_H_

#include <string_view>

#include <nlohmann/json.hpp>
#include "tensorstore/kvstore/spec.h"

namespace tensorstore {
namespace internal {

// Collect all metrics with the `prefix` and return a json array of them
//
// When `id` is set, an object {identifier: id} will be aappended
::nlohmann::json CollectMetricsToJson(std::string id, std::string_view prefix);

// Return the json object loading from the metric json specified by kvstore_spec
//
// Return empty json object if the metric file doesn't exist
::nlohmann::json ReadMetricCollectionFromKvstore(
    const kvstore::Spec& kvstore_spec);

// Write `all_metrics` to `kvstore_spec` if it sets properly and return true.
//
// return False if upload fails
absl::Status WriteMetricCollectionToKvstore(::nlohmann::json all_metrics,
                                            const kvstore::Spec& kvstore_spec,
                                            bool final_collect = true);

// Print out metrics to stdout, sorted by keys
void DumpMetrics(std::string_view prefix);

};  // namespace internal
};  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BENCHMARK_METRIC_UTILS_H_
