// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_METRICS_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_METRICS_H_

#include <stddef.h>
#include <stdint.h>

#include <array>

#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/domain_field.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"

namespace tensorstore {
namespace internal_http {

struct HttpStatusDomain {
  static constexpr std::array<int, 34> kValues = {
      100, 101, 103, 200, 201, 202, 204, 206, 207, 301, 302, 304,
      307, 308, 400, 401, 403, 404, 405, 406, 408, 409, 410, 412,
      415, 416, 418, 429, 451, 500, 501, 502, 503, 504,
  };

  // FIND_SEED --type=int32_t --table_size=42
  // 100 101 103 200 201 202 204 206 207 301 302 304 307 308 \
  // 400 401 403 404 405 406 408 409 410 412 415 416 418 429 \
  // 451 500 501 502 503 504
  static constexpr uint32_t kSeed = 5799507;
  static constexpr size_t kTableSize = 42;
};

struct HttpMetrics {
  internal_metrics::Counter<int64_t> request_started;
  internal_metrics::Counter<int64_t> request_completed;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer> request_bytes;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>
      request_header_bytes;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer> response_bytes;
  internal_metrics::Gauge<int64_t> active;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer> total_time_ms;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>
      first_byte_latency_us;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer> poll_time_ns;
  internal_metrics::Counter<int64_t,
                            internal_metrics::DomainField<HttpStatusDomain>>
      response_codes;
};

/// Returns the shared `/tensorstore/http/*` metric collection registered with
/// `GetMetricRegistry()`.
HttpMetrics& GetHttpMetrics();

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_METRICS_H_
