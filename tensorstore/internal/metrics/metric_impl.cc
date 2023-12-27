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

#include "tensorstore/internal/metrics/metric_impl.h"

#include <stddef.h>

#include <atomic>

namespace tensorstore {
namespace internal_metrics {

size_t MetricThreadCounter() {
  static std::atomic<size_t> thread_counter{0};
  return thread_counter.fetch_add(1);
}

}  // namespace internal_metrics
}  // namespace tensorstore
