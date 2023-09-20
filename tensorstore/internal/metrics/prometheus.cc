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

#include "tensorstore/internal/metrics/prometheus.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_metrics {
namespace {

static inline constexpr internal::AsciiSet kDigit{"0123456789"};

// Metric names MUST adhere to the regex [a-zA-Z_:]([a-zA-Z0-9_:])*.
static inline constexpr internal::AsciiSet kMetricFirst{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_:"};

// Label names MUST adhere to the regex [a-zA-Z_]([a-zA-Z0-9_])*.
static inline constexpr internal::AsciiSet kLabelFirst{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "_"};

// Label values MAY be any sequence of UTF-8 characters, however they need to be
// uri-safe.
static inline constexpr internal::AsciiSet kValueUnreserved{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.~()"};

bool IsLegalPrometheusLabel(std::string_view label) {
  if (label.empty() || !kLabelFirst.Test(label[0])) return false;
  for (char c : label) {
    if (!kLabelFirst.Test(c) && !kDigit.Test(c)) return false;
  }
  return true;
}

absl::Status AppendLabelValue(std::string* url, std::string_view label,
                              std::string_view value) {
  if (!IsLegalPrometheusLabel(label)) {
    return absl::InvalidArgumentError("");
  }
  if (value.empty()) {
    absl::StrAppend(url, "/", label, "@base64/=");
  }
  for (char c : value) {
    if (!kValueUnreserved.Test(c)) {
      absl::StrAppend(url, "/", label, "@base64/",
                      absl::WebSafeBase64Escape(value));
      return absl::OkStatus();
    }
  }
  absl::StrAppend(url, "/", label, "/", value);
  return absl::OkStatus();
}

// Returns a prometheus string, which is one which begins with the AsciiSet
// first, and where all the non numeric, non-first characters are replaced by _
std::string AsPrometheusString(std::string_view in, internal::AsciiSet first) {
  while (!in.empty() && !first.Test(in[0])) {
    in = in.substr(1);
  }
  while (!in.empty() && !first.Test(in[in.size() - 1]) &&
         !kDigit.Test(in[in.size() - 1])) {
    in = in.substr(0, in.size() - 1);
  }
  std::string raw(in);
  for (char& c : raw) {
    if (!first.Test(c) && !kDigit.Test(c)) c = '_';
  }
  return raw;
}

struct PrometheusValueLine {
  const std::string& metric_name;
  const char* suffix;
  const std::string& label_str;

  std::string operator()(int64_t x) {
    return absl::StrCat(metric_name, suffix, label_str.empty() ? "" : "{",
                        label_str, label_str.empty() ? "" : "} ", x);
  }
  std::string operator()(double x) {
    return absl::StrCat(metric_name, suffix, label_str.empty() ? "" : "{",
                        label_str, label_str.empty() ? "" : "} ", x);
  }
  std::string operator()(const std::string& x) { return {}; }
  std::string operator()(std::monostate) { return {}; }
};

}  // namespace

Result<internal_http::HttpRequest> BuildPrometheusPushRequest(
    const PushGatewayConfig& config) {
  if (config.job.empty()) {
    return absl::InvalidArgumentError("PushGatewayConfig bad job");
  }
  if (!absl::StartsWith(config.host, "http://") &&
      !absl::StartsWith(config.host, "https://")) {
    return absl::InvalidArgumentError("PushGatewayConfig bad host");
  }

  std::string url = config.host;
  if (!absl::EndsWith(url, "/")) {
    absl::StrAppend(&url, "/metrics");
  } else {
    absl::StrAppend(&url, "metrics");
  }

  TENSORSTORE_RETURN_IF_ERROR(AppendLabelValue(&url, "job", config.job));

  if (!config.instance.empty()) {
    TENSORSTORE_RETURN_IF_ERROR(
        AppendLabelValue(&url, "instance", config.instance));
  }

  for (const auto& [k, v] : config.additional_labels) {
    if (absl::EqualsIgnoreCase("job", k) ||
        absl::EqualsIgnoreCase("instance", k)) {
      return absl::InvalidArgumentError(
          "PushGatewayConfig additional_labels cannot contain job or instance");
    }
    TENSORSTORE_RETURN_IF_ERROR(AppendLabelValue(&url, k, v));
  }

  return internal_http::HttpRequestBuilder("PUT", std::move(url))
      .BuildRequest();
}

void PrometheusExpositionFormat(
    const CollectedMetric& metric,
    absl::FunctionRef<void(std::string)> handle_line) {
  /// Construct metric_name.
  std::string metric_name =
      AsPrometheusString(metric.metric_name, kMetricFirst);
  if (metric_name.empty()) return;

  std::vector<std::string> prometheus_fields;
  prometheus_fields.reserve(metric.field_names.size());
  for (size_t i = 0; i < metric.field_names.size(); ++i) {
    prometheus_fields.push_back(
        AsPrometheusString(metric.field_names[i], kLabelFirst));
  }

  auto build_label_str = [&](auto& v) -> std::string {
    assert(metric.field_names.size() == v.fields.size());
    if (v.fields.empty()) return {};
    std::string label_str;
    for (size_t i = 0; i < metric.field_names.size(); ++i) {
      absl::StrAppend(&label_str, i == 0 ? "" : ", ", prometheus_fields[i],
                      "=\"", absl::CEscape(v.fields[i]), "\"");
    }
    return label_str;
  };

  if (!metric.values.empty()) {
    std::string line;
    for (const auto& v : metric.values) {
      // Build labels for values.
      std::string label_str = build_label_str(v);
      line =
          std::visit(PrometheusValueLine{metric_name, " ", label_str}, v.value);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
      line = std::visit(PrometheusValueLine{metric_name, "_max ", label_str},
                        v.max_value);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
    }
  }
  if (!metric.histograms.empty()) {
    std::string line;
    for (const auto& v : metric.histograms) {
      // Build labels for values.
      std::string label_str = build_label_str(v);

      // Representation of a Histogram metric.
      struct Histogram {
        std::vector<int64_t> buckets;
      };

      line = PrometheusValueLine{metric_name, "_mean ", label_str}(v.mean);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
      line = PrometheusValueLine{metric_name, "_count ", label_str}(v.count);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
      line = PrometheusValueLine{metric_name, "_variance ",
                                 label_str}(v.sum_of_squared_deviation);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
      line = PrometheusValueLine{metric_name, "_sum ",
                                 label_str}(v.mean * v.count);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
      size_t end = v.buckets.size();
      while (end > 0 && v.buckets[end - 1] == 0) --end;

      for (size_t i = 0; i < end; i++) {
        std::string bucket_labels = absl::StrCat(
            label_str, label_str.empty() ? "" : ", ", "le=\"", i, "\"");
        line = PrometheusValueLine{metric_name, "_bucket ",
                                   bucket_labels}(v.buckets[i]);
        if (!line.empty()) {
          handle_line(std::move(line));
        }
      }
      // Add +Inf bucket
      std::string bucket_labels =
          absl::StrCat(label_str, label_str.empty() ? "" : ", ", "le=\"+Inf\"");
      line =
          PrometheusValueLine{metric_name, "_bucket ", bucket_labels}(v.count);
      if (!line.empty()) {
        handle_line(std::move(line));
      }
    }
  }
}

}  // namespace internal_metrics
}  // namespace tensorstore
