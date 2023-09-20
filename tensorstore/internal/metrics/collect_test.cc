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

#include "tensorstore/internal/metrics/collect.h"

#include <stdint.h>

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"

namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::CollectedMetricToJson;
using ::tensorstore::internal_metrics::FormatCollectedMetric;
using ::tensorstore::internal_metrics::IsCollectedMetricNonZero;
using ::testing::ElementsAre;
using ::testing::Pair;

TEST(CollectTest, IsCollectedMetricNonZero) {
  CollectedMetric metric;
  // Metadata does not make a metric non-zero.
  metric.metric_name = "metric_name";
  metric.field_names.push_back("field_name");
  metric.metadata.description = "description";
  metric.tag = "tag";
  EXPECT_FALSE(IsCollectedMetricNonZero(metric));

  // Field metadata does not make a metric non-zero.
  metric.histograms.push_back(CollectedMetric::Histogram{});
  auto& h = metric.histograms.back();
  h.fields.push_back("hh");
  h.count = 0;
  EXPECT_FALSE(IsCollectedMetricNonZero(metric));

  // A single histogram count makes h non-zero.
  h.count = 1;
  EXPECT_TRUE(IsCollectedMetricNonZero(metric));

  // Field metadata does not make a metric non-zero.
  metric.histograms.clear();
  metric.values.push_back(CollectedMetric::Value{});
  auto& v = metric.values.back();
  v.fields.push_back("vv");

  // A value makes v non zero.
  v.value = int64_t{1};
  EXPECT_TRUE(IsCollectedMetricNonZero(metric));
  v.value = std::monostate{};

  // A max_value makes v non zero.
  v.max_value = int64_t{1};
  EXPECT_TRUE(IsCollectedMetricNonZero(metric));
  v.max_value = std::monostate{};
}

TEST(CollectTest, FormatCollectedMetric) {
  auto format_lines = [](const CollectedMetric& metric) {
    std::vector<std::pair<bool, std::string>> lines;
    FormatCollectedMetric(
        metric, [&](bool has_value, std::string formatted_line) {
          lines.push_back(std::make_pair(has_value, std::move(formatted_line)));
        });
    return lines;
  };

  EXPECT_THAT(format_lines({}), testing::IsEmpty());

  CollectedMetric metric;
  metric.metric_name = "metric_name";
  metric.field_names.push_back("field_name");
  metric.metadata.description = "description";
  metric.tag = "tag";

  {
    metric.values.push_back(CollectedMetric::Value{});
    auto& v = metric.values.back();
    v.fields.push_back("vv");
    v.value = int64_t{1};

    EXPECT_THAT(format_lines(metric),
                ElementsAre(Pair(true, "metric_name<field_name>[vv]=1")));
  }

  {
    metric.values.clear();
    metric.histograms.push_back(CollectedMetric::Histogram{});
    auto& h = metric.histograms.back();
    h.fields.push_back("hh");
    h.count = 1;

    EXPECT_THAT(format_lines(metric),
                ElementsAre(Pair(true,
                                 "metric_name<field_name>[hh]={count=1 "
                                 "mean=0 buckets=[]}")));
  }
}

TEST(CollectTest, CollectedMetricToJson) {
  EXPECT_THAT(
      CollectedMetricToJson({}),
      MatchesJson({{"name", ""}, {"values", nlohmann::json::array_t()}}));

  CollectedMetric metric;
  metric.metric_name = "metric_name";
  metric.field_names.push_back("field_name");
  metric.metadata.description = "description";
  metric.tag = "tag";

  {
    metric.values.push_back(CollectedMetric::Value{});
    auto& v = metric.values.back();
    v.fields.push_back("vv");
    v.value = int64_t{1};
    EXPECT_THAT(CollectedMetricToJson(metric),
                MatchesJson({{"name", "metric_name"},
                             {"values",
                              {{
                                  {"value", 1},
                                  {"field_name", "vv"},
                              }}}}));
  }

  {
    metric.values.clear();
    metric.histograms.push_back(CollectedMetric::Histogram{});
    auto& h = metric.histograms.back();
    h.fields.push_back("hh");
    h.count = 1;

    EXPECT_THAT(CollectedMetricToJson(metric),
                MatchesJson({{"name", "metric_name"},
                             {"values",
                              {{
                                  {"count", 1},
                                  {"field_name", "hh"},
                                  {"mean", 0.0},
                                  {"sum_of_squared_deviation", 0.0},
                              }}}}));
  }
}

}  // namespace
