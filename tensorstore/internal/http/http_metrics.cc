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

#include "tensorstore/internal/http/http_metrics.h"

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registry.h"

namespace tensorstore {
namespace internal_http {
namespace {

using ::tensorstore::internal_metrics::GetMetricRegistry;
using ::tensorstore::internal_metrics::MetricMetadata;
using ::tensorstore::internal_metrics::Units;

ABSL_CONST_INIT HttpMetrics g_http_metrics;
ABSL_CONST_INIT absl::once_flag g_register_once;

void EnsureHttpMetricsRegistered() {
  absl::call_once(g_register_once, []() {
    auto& r = GetMetricRegistry();
    r.Register(&g_http_metrics.request_started,
               MetricMetadata("/tensorstore/http/request_started",
                              "HTTP requests started"));
    r.Register(&g_http_metrics.request_completed,
               MetricMetadata("/tensorstore/http/request_completed",
                              "HTTP requests completed"));
    r.Register(&g_http_metrics.request_bytes,
               MetricMetadata("/tensorstore/http/request_bytes",
                              "HTTP request bytes transmitted", Units::kBytes));
    r.Register(&g_http_metrics.request_header_bytes,
               MetricMetadata("/tensorstore/http/request_header_bytes",
                              "HTTP request bytes transmitted", Units::kBytes));
    r.Register(&g_http_metrics.response_bytes,
               MetricMetadata("/tensorstore/http/response_bytes",
                              "HTTP response bytes received", Units::kBytes));
    r.Register(&g_http_metrics.active,
               MetricMetadata("/tensorstore/http/active",
                              "HTTP requests considered active"));
    r.Register(&g_http_metrics.total_time_ms,
               MetricMetadata("/tensorstore/http/total_time_ms",
                              "HTTP total latency (ms)", Units::kMilliseconds));
    r.Register(&g_http_metrics.first_byte_latency_us,
               MetricMetadata("/tensorstore/http/first_byte_latency_us",
                              "HTTP first byte received latency (us)",
                              Units::kMicroseconds));
    r.Register(&g_http_metrics.poll_time_ns,
               MetricMetadata("/tensorstore/http/http_poll_time_ns",
                              "HTTP time spent in curl_multi_poll (ns)",
                              Units::kNanoseconds));
    r.Register(&g_http_metrics.response_codes,
               MetricMetadata("/tensorstore/http/response_codes",
                              "HTTP response status code counts")
                   .WithFieldNames({"code"}));
  });
}

const struct HttpMetricsInitializer {
  HttpMetricsInitializer() { EnsureHttpMetricsRegistered(); }
} g_http_metrics_initializer;

}  // namespace

HttpMetrics& GetHttpMetrics() {
  EnsureHttpMetricsRegistered();
  return g_http_metrics;
}

}  // namespace internal_http
}  // namespace tensorstore
