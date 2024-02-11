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

#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"

namespace {

using ::tensorstore::internal_metrics::BuildPrometheusPushRequest;
using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::PrometheusExpositionFormat;
using ::tensorstore::internal_metrics::PushGatewayConfig;

TEST(PrometheusTest, BuildPrometheusPushRequest) {
  auto request = BuildPrometheusPushRequest(
      PushGatewayConfig{"http://localhost:8080", "my_job", "1", {}});

  EXPECT_TRUE(request.has_value());
  EXPECT_EQ("http://localhost:8080/metrics/job/my_job/instance/1",
            request->url);
}

TEST(PrometheusTest, PrometheusExpositionFormat) {
  auto format_lines = [](const CollectedMetric& metric) {
    std::vector<std::string> lines;
    PrometheusExpositionFormat(
        metric, [&](std::string line) { lines.push_back(std::move(line)); });
    return lines;
  };

  CollectedMetric metric;
  metric.metric_name = "metric_name";
  metric.field_names.push_back("field_name");
  metric.metadata.description = "description";
  metric.tag = "tag";
  EXPECT_THAT(format_lines(metric), ::testing::IsEmpty());

  metric.histograms.push_back(CollectedMetric::Histogram{});
  auto& h = metric.histograms.back();
  h.fields.push_back("hh");
  h.count = 1;
  h.mean = 1;
  h.sum_of_squared_deviation = 1;
  h.buckets.push_back(0);
  h.buckets.push_back(1);

  metric.values.push_back(CollectedMetric::Value{});
  auto& v = metric.values.back();
  v.fields.push_back("vv");
  v.value = int64_t{1};
  v.max_value = int64_t{2};
  EXPECT_THAT(format_lines(metric),
              ::testing::ElementsAre(
                  "metric_name {field_name=\"vv\"} 1",
                  "metric_name_max {field_name=\"vv\"} 2",
                  "metric_name_mean {field_name=\"hh\"} 1",
                  "metric_name_count {field_name=\"hh\"} 1",
                  "metric_name_variance {field_name=\"hh\"} 1",
                  "metric_name_sum {field_name=\"hh\"} 1",
                  "metric_name_bucket {field_name=\"hh\", le=\"0\"} 0",
                  "metric_name_bucket {field_name=\"hh\", le=\"1\"} 1",
                  "metric_name_bucket {field_name=\"hh\", le=\"+Inf\"} 1"));
}

}  // namespace
