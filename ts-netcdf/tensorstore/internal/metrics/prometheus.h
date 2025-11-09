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

#ifndef TENSORSTORE_INTERNAL_METRICS_PROMETHEUS_H_
#define TENSORSTORE_INTERNAL_METRICS_PROMETHEUS_H_

#include <map>
#include <string>

#include "absl/functional/function_ref.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_metrics {

// Utility to write prometheus metrics to a push gateway
// https://github.com/prometheus/pushgateway
//
// Metric names MUST adhere to the regex [a-zA-Z_:]([a-zA-Z0-9_:])*.
// Label names MUST adhere to the regex [a-zA-Z_]([a-zA-Z0-9_])*.
// Label values MAY be any sequence of UTF-8 characters.
struct PushGatewayConfig {
  std::string host;

  // instance_labels identify the job in the prometheus pushgateway.
  // instance_labels are a map from label name to label value.
  // labels must adhere to the regex: [] to the spec
  std::string job;
  std::string instance;
  std::map<std::string, std::string> additional_labels;
};

Result<internal_http::HttpRequest> BuildPrometheusPushRequest(
    const PushGatewayConfig& config);

// Format metrics in prometheus exposition format.
//
// https://github.com/prometheus/docs/blob/main/content/docs/instrumenting/exposition_formats.md
//
// This format includes one metric per line, where the line format is:
// metric_name [
//   "{" label_name "=" `"` label_value `"` { "," label_name "=" `"` label_value
//   `"` } [ "," ] "}"
// ] value [ timestamp ]
//
void PrometheusExpositionFormat(
    const CollectedMetric& metric,
    absl::FunctionRef<void(std::string)> handle_line);

}  // namespace internal_metrics
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_METRICS_PROMETHEUS_H_
