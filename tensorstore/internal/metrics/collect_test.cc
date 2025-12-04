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
#include "absl/types/compare.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/testing/json_gtest.h"

namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::CollectedMetricDelta;
using ::tensorstore::internal_metrics::CollectedMetricToJson;
using ::tensorstore::internal_metrics::CompareByName;
using ::tensorstore::internal_metrics::FormatCollectedMetric;
using ::tensorstore::internal_metrics::IsCollectedMetricNonZero;
using ::tensorstore::internal_metrics::Units;
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
  metric.metadata.units = Units::kBytes;
  metric.tag = "tag";
  metric.histogram_labels = {"0", "3", "Inf"};

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
    h.mean = 1;
    h.sum_of_squared_deviation = 1;
    h.buckets.push_back(0);
    h.buckets.push_back(1);

    EXPECT_THAT(CollectedMetricToJson(metric),
                MatchesJson({{"name", "metric_name"},
                             {"values",
                              {{
                                  {"count", 1},
                                  {"field_name", "hh"},
                                  {"mean", 1.0},
                                  {"sum_of_squared_deviation", 1.0},
                                  {"0", 0},
                                  {"3", 1},
                              }}}}));
  }
}

TEST(CollectTest, CompareCollectedMetricByName) {
  CollectedMetric a, b;
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::equivalent);
  a.metric_name = "a";
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::greater);
  b.metric_name = "a";
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::equivalent);
  b.tag = "a";
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::less);
  a.tag = "a";
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::equivalent);
  a.field_names.push_back("b");
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::greater);
  b.field_names.push_back("a");
  EXPECT_THAT(CompareByName(a, b), absl::weak_ordering::greater);
}

// TODO: Add tests for more complex CollectedMetricDelta cases.
TEST(CollectTest, CollectedMetricDelta_Values) {
  CollectedMetric a;
  a.metric_name = "a";
  a.field_names.push_back("fn");

  CollectedMetric b = a;
  a.values.push_back({{"a"}, int64_t{10}, int64_t{100}});
  a.values.push_back({{"b"}, int64_t{1}, int64_t{20}});
  a.values.push_back({{"c"}, int64_t{1}, int64_t{20}});
  b.values.push_back({{"a"}, int64_t{12}, int64_t{110}});
  b.values.push_back({{"b"}, int64_t{2}, int64_t{10}});
  b.values.push_back({{"d"}, int64_t{2}, int64_t{10}});

  CollectedMetric c = CollectedMetricDelta(a, b);

  EXPECT_THAT(CollectedMetricToJson(c),
              MatchesJson({{"name", "a"},
                           {
                               "values",
                               {{
                                    {"fn", "a"},
                                    {"max_value", 10},
                                    {"value", 2},
                                },
                                {
                                    {"fn", "b"},
                                    {"max_value", -10},
                                    {"value", 1},
                                },
                                {
                                    {"fn", "c"},
                                    {"max_value", -20},
                                    {"value", -1},
                                },
                                {
                                    {"fn", "d"},
                                    {"max_value", 10},
                                    {"value", 2},
                                }},
                           }}));
}

TEST(CollectTest, CollectedMetricDelta_Histograms) {
  CollectedMetric a;
  a.metric_name = "a";
  a.histogram_labels = {"0", "3", "Inf"};

  CollectedMetric b = a;
  a.histograms.push_back(CollectedMetric::Histogram{});
  {
    auto& h = a.histograms.back();
    h.count = 2;
    h.mean = 1;
    h.sum_of_squared_deviation = 1;
    h.buckets.push_back(0);
    h.buckets.push_back(1);
  }
  b.histograms.push_back(a.histograms.back());
  {
    auto& h = a.histograms.back();
    h.count = 2;
    h.mean = 9;
    h.sum_of_squared_deviation = 6;
    h.buckets.clear();
    h.buckets.push_back(6);
    h.buckets.push_back(12);
  }

  CollectedMetric c = CollectedMetricDelta(a, b);

  EXPECT_THAT(CollectedMetricToJson(c),
              MatchesJson({{"name", "a"},
                           {"values",
                            {{
                                {"count", 0},
                                {"mean", -8.0},
                                {"sum_of_squared_deviation", -5.0},
                                {"0", -6},
                                {"3", -11},
                            }}}}));
}

TEST(CollectTest, CollectedMetricDeltaSizeMismatch) {
  CollectedMetric a, b;
  a.metric_name = "a";
  b.metric_name = "a";
  a.values.push_back({{}, int64_t{10}, int64_t{100}});
  b.values.push_back({{}, int64_t{12}, int64_t{110}});
  b.values.push_back({{}, int64_t{13}, int64_t{111}});
  a.histograms.push_back({{}, 1, 2.0, 3.0, {4, 5}});
  b.histograms.push_back({{}, 6, 8.0, 10.0, {11, 12, 13}});

  CollectedMetric c = CollectedMetricDelta(a, b);
  EXPECT_EQ("a", c.metric_name);

  ASSERT_EQ(2, c.values.size());
  EXPECT_THAT(c.values[0].value, 2);
  EXPECT_THAT(c.values[0].max_value, 10);
  EXPECT_THAT(c.values[1].value, 13);
  EXPECT_THAT(c.values[1].max_value, 111);

  ASSERT_EQ(1, c.histograms.size());
  EXPECT_EQ(5, c.histograms[0].count);
  EXPECT_EQ(6.0, c.histograms[0].mean);
  EXPECT_EQ(7.0, c.histograms[0].sum_of_squared_deviation);
  ASSERT_EQ(3, c.histograms[0].buckets.size());
  EXPECT_EQ(7, c.histograms[0].buckets[0]);
  EXPECT_EQ(7, c.histograms[0].buckets[1]);
  EXPECT_EQ(13, c.histograms[0].buckets[2]);
}

}  // namespace
