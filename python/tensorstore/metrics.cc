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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <stdint.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/cord.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/prometheus.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_python {
namespace {

std::vector<::nlohmann::json> CollectMatchingMetrics(
    std::string metric_prefix, bool include_zero_metrics) {
  std::vector<::nlohmann::json> lines;

  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix)) {
    if (include_zero_metrics ||
        internal_metrics::IsCollectedMetricNonZero(metric)) {
      lines.push_back(internal_metrics::CollectedMetricToJson(metric));
    }
  }

  std::sort(std::begin(lines), std::end(lines));
  return lines;
}

std::vector<std::string> CollectPrometheusFormatMetrics(
    std::string metric_prefix) {
  std::vector<std::string> lines;
  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix)) {
    PrometheusExpositionFormat(
        metric, [&](std::string line) { lines.push_back(std::move(line)); });
  }
  return lines;
}

Future<uint32_t> PushMetricsToPrometheus(std::string pushgateway,
                                         std::string job, std::string instance,
                                         std::string metric_prefix) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto request,
      BuildPrometheusPushRequest(
          internal_metrics::PushGatewayConfig{pushgateway, job, instance, {}}));

  // Collect the metrics.
  absl::Cord payload;
  for (const auto& metric :
       internal_metrics::GetMetricRegistry().CollectWithPrefix(metric_prefix)) {
    PrometheusExpositionFormat(metric, [&](std::string line) {
      line.append("\n");
      payload.Append(std::move(line));
    });
  }

  return MapFuture(
      InlineExecutor{},
      [](const Result<internal_http::HttpResponse>& result)
          -> Result<uint32_t> {
        TENSORSTORE_RETURN_IF_ERROR(result.status());
        return result->status_code;
      },
      internal_http::GetDefaultHttpTransport()->IssueRequest(
          request, internal_http::IssueRequestOptions(std::move(payload))));
}

}  // namespace

void RegisterMetricBindings(pybind11::module_ m, Executor defer) {
  m.def("experimental_collect_matching_metrics",
        &internal_python::CollectMatchingMetrics,
        pybind11::arg("metric_prefix") = "",
        pybind11::arg("include_zero_metrics") = false, R"(
Collects metrics with a matching prefix.

Args:
  metric_prefix: Prefix of the metric names to collect.
  include_zero_metrics: Indicate whether zero-valued metrics are included.

Returns:
  :py:obj:`list` of a :py:obj:`dict` of metrics.

Group:
  Experimental
)");

  m.def("experimental_collect_prometheus_format_metrics",
        &internal_python::CollectPrometheusFormatMetrics,
        pybind11::arg("metric_prefix") = "", R"(
Collects metrics in prometheus exposition format.
See: https://prometheus.io/docs/instrumenting/exposition_formats/

Args:
  metric_prefix: Prefix of the metric names to collect.

Returns:
  :py:obj:`list` of a :py:obj:`str` of prometheus exposition format metrics.

Group:
  Experimental
)");

  m.def("experimental_push_metrics_to_prometheus",
        &internal_python::PushMetricsToPrometheus,
        pybind11::arg("pushgateway") = "", pybind11::arg("job") = "",
        pybind11::arg("instance") = "", pybind11::arg("metric_prefix") = "",
        R"(
Publishes metrics to the prometheus pushgateway.
See: https://github.com/prometheus/pushgateway

Args:
  pushgateway: prometheus pushgateway url, like 'http://localhost:1234/'
  job: prometheus job name
  instance: prometheus instance identifier
  metric_prefix: Prefix of the metric names to publish.

Returns:
  A future with the response status code.

Group:
  Experimental
)");
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterMetricBindings, /*priority=*/-1);
}

}  // namespace internal_python
}  // namespace tensorstore
