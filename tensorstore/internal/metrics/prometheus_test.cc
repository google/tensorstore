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

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/domain_field.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registry.h"

namespace {

using ::tensorstore::internal_metrics::BuildPrometheusPushRequest;
using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::Counter;
using ::tensorstore::internal_metrics::GetMetricRegistry;
using ::tensorstore::internal_metrics::MetricMetadata;
using ::tensorstore::internal_metrics::PrometheusExpositionFormat;
using ::tensorstore::internal_metrics::PushGatewayConfig;
using ::tensorstore::internal_metrics::Units;

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
  metric.metadata.units = Units::kBytes;
  metric.tag = "tag";
  EXPECT_THAT(format_lines(metric), ::testing::IsEmpty());

  metric.values.push_back(CollectedMetric::Value{});
  auto& v = metric.values.back();
  v.fields.push_back("vv");
  v.value = int64_t{1};
  v.max_value = int64_t{2};
  EXPECT_THAT(format_lines(metric),
              ::testing::ElementsAre("# HELP metric_name description",  //
                                     "# UNIT metric_name bytes",        //
                                     "metric_name {field_name=\"vv\"} 1",
                                     "metric_name_max {field_name=\"vv\"} 2"));
}

TEST(PrometheusTest, PrometheusExpositionFormat_Histogram) {
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
  metric.metadata.units = Units::kBytes;
  metric.tag = "tag";
  metric.histogram_labels = {"0", "3", "Inf"};

  metric.histograms.push_back(CollectedMetric::Histogram{});
  auto& h = metric.histograms.back();
  h.fields.push_back("hh");
  h.count = 1;
  h.mean = 1;
  h.sum_of_squared_deviation = 1;
  h.buckets.push_back(0);
  h.buckets.push_back(1);

  EXPECT_THAT(format_lines(metric),
              ::testing::ElementsAre(
                  "# HELP metric_name description",  //
                  "# UNIT metric_name bytes",        //
                  "# TYPE metric_name histogram",    //
                  "metric_name_mean {field_name=\"hh\"} 1",
                  "metric_name_count {field_name=\"hh\"} 1",
                  "metric_name_variance {field_name=\"hh\"} 1",
                  "metric_name_sum {field_name=\"hh\"} 1",
                  "metric_name_bucket {field_name=\"hh\", le=\"0\"} 0",
                  "metric_name_bucket {field_name=\"hh\", le=\"3\"} 1",
                  "metric_name_bucket {field_name=\"hh\", le=\"+Inf\"} 1"));
}

struct TestMethodDomain {
  static constexpr std::array<std::string_view, 2> kValues = {"Read", "Write"};
  // FIND_SEED Read Write
  static constexpr uint32_t kSeed = 0;
  static constexpr size_t kTableSize = 2;
};
using MethodField =
    tensorstore::internal_metrics::DomainField<TestMethodDomain>;

enum class TestTask {
  kTaskA = 1,
  kTaskB = 2,
};
struct TestTaskDomain {
  static constexpr std::array<TestTask, 2> kValues = {TestTask::kTaskA,
                                                      TestTask::kTaskB};
};
using TaskField = tensorstore::internal_metrics::DomainField<TestTaskDomain>;

TEST(PrometheusTest, DomainFieldsExport) {
  auto format_lines = [](const CollectedMetric& metric) {
    std::vector<std::string> lines;
    PrometheusExpositionFormat(
        metric, [&](std::string line) { lines.push_back(std::move(line)); });
    return lines;
  };

  static Counter<int64_t, MethodField, TaskField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/test/prometheus_domain_fields",
                                 "Domain metrics", {"method", "task"}));
    return true;
  }();

  counter.Increment("Read", TestTask::kTaskA);
  counter.IncrementBy(3, "Write", TestTask::kTaskB);
  // Increment an invalid value to populate the overflow cell
  counter.Increment("InvalidMethod", TestTask::kTaskA);

  auto metric =
      GetMetricRegistry().Collect("/tensorstore/test/prometheus_domain_fields");
  ASSERT_TRUE(metric.has_value());

  EXPECT_THAT(
      format_lines(*metric),
      ::testing::UnorderedElementsAre(
          "# HELP tensorstore_test_prometheus_domain_fields Domain metrics",
          "tensorstore_test_prometheus_domain_fields "
          "{method=\"Read\", task=\"1\"} 1",
          "tensorstore_test_prometheus_domain_fields "
          "{method=\"Read\", task=\"2\"} 0",
          "tensorstore_test_prometheus_domain_fields "
          "{method=\"Write\", task=\"1\"} 0",
          "tensorstore_test_prometheus_domain_fields "
          "{method=\"Write\", task=\"2\"} 3",
          "tensorstore_test_prometheus_domain_fields "
          "{method=\"\", task=\"\"} 1"));
}

}  // namespace
