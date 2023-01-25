// Copyright 2022 The TensorStore Authors
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

#include <algorithm>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/metrics/value.h"

namespace {

using ::tensorstore::internal_metrics::Counter;
using ::tensorstore::internal_metrics::DefaultBucketer;
using ::tensorstore::internal_metrics::Gauge;
using ::tensorstore::internal_metrics::GetMetricRegistry;
using ::tensorstore::internal_metrics::Histogram;
using ::tensorstore::internal_metrics::Value;

TEST(MetricTest, CounterInt) {
  auto& counter = Counter<int64_t>::New("/tensorstore/counter1", "A metric");
  counter.Increment();
  counter.IncrementBy(2);

  EXPECT_EQ(3, counter.Get());

  {
    auto metric = counter.Collect();

    EXPECT_EQ("/tensorstore/counter1", metric.metric_name);
    EXPECT_TRUE(metric.field_names.empty());
    ASSERT_EQ(1, metric.counters.size());
    EXPECT_TRUE(metric.counters[0].fields.empty());
    EXPECT_EQ(3, std::get<int64_t>(metric.counters[0].value));
  }
  {
    auto metric = GetMetricRegistry().Collect("/tensorstore/counter1");
    ASSERT_TRUE(metric.has_value());

    EXPECT_EQ("/tensorstore/counter1", metric->metric_name);
    EXPECT_TRUE(metric->field_names.empty());
    ASSERT_EQ(1, metric->counters.size());
    EXPECT_TRUE(metric->counters[0].fields.empty());
    EXPECT_EQ(3, std::get<int64_t>(metric->counters[0].value));
  }
}

TEST(MetricTest, CounterDouble) {
  auto& counter = Counter<double>::New("/tensorstore/counter2", "A metric");
  counter.Increment();
  counter.IncrementBy(2);

  auto metric = counter.Collect();
  EXPECT_EQ("/tensorstore/counter2", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.counters.size());
  EXPECT_TRUE(metric.counters[0].fields.empty());
  EXPECT_EQ(3, std::get<double>(metric.counters[0].value));
}

TEST(MetricTest, CounterIntFields) {
  auto& counter = Counter<int64_t, std::string>::New("/tensorstore/counter3",
                                                     "field1", "A metric");
  counter.Increment("a");
  counter.IncrementBy(2, "b");

  EXPECT_EQ(1, counter.Get("a"));
  EXPECT_EQ(2, counter.Get("b"));

  auto metric = counter.Collect();
  EXPECT_EQ("/tensorstore/counter3", metric.metric_name);
  EXPECT_THAT(metric.field_names, ::testing::ElementsAre("field1"));

  ASSERT_EQ(2, metric.counters.size());
  std::sort(metric.counters.begin(), metric.counters.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_THAT(metric.counters[0].fields, ::testing::ElementsAre("a"));
  EXPECT_EQ(1, std::get<int64_t>(metric.counters[0].value));
  EXPECT_THAT(metric.counters[1].fields, ::testing::ElementsAre("b"));
  EXPECT_EQ(2, std::get<int64_t>(metric.counters[1].value));
}

TEST(MetricTest, CounterDoubleFields) {
  auto& counter =
      Counter<double, int>::New("/tensorstore/counter4", "field1", "A metric");
  counter.Increment(1);
  counter.IncrementBy(2, 2);

  auto metric = counter.Collect();

  EXPECT_EQ("/tensorstore/counter4", metric.metric_name);
  EXPECT_THAT(metric.field_names, ::testing::ElementsAre("field1"));

  ASSERT_EQ(2, metric.counters.size());
  std::sort(metric.counters.begin(), metric.counters.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_THAT(metric.counters[0].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(1, std::get<double>(metric.counters[0].value));
  EXPECT_THAT(metric.counters[1].fields, ::testing::ElementsAre("2"));
  EXPECT_EQ(2, std::get<double>(metric.counters[1].value));
}

TEST(MetricTest, GaugeInt) {
  auto& gauge = Gauge<int64_t>::New("/tensorstore/gauge1", "A metric");
  gauge.Set(3);
  gauge.Increment();
  gauge.IncrementBy(2);

  EXPECT_EQ(6, gauge.Get());
  EXPECT_EQ(6, gauge.GetMax());

  auto metric = gauge.Collect();

  EXPECT_EQ("/tensorstore/gauge1", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.gauges.size());
  EXPECT_TRUE(metric.gauges[0].fields.empty());
  EXPECT_EQ(6, std::get<int64_t>(metric.gauges[0].value));
}

TEST(MetricTest, GaugeDouble) {
  auto& gauge = Gauge<double>::New("/tensorstore/gauge2", "A metric");
  gauge.Set(3);
  gauge.Increment();
  gauge.IncrementBy(2);

  auto metric = gauge.Collect();

  EXPECT_EQ("/tensorstore/gauge2", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.gauges.size());
  EXPECT_TRUE(metric.gauges[0].fields.empty());
  EXPECT_EQ(6, std::get<double>(metric.gauges[0].value));
}

TEST(MetricTest, GaugeIntFields) {
  auto& gauge = Gauge<int64_t, std::string>::New("/tensorstore/gauge3",
                                                 "field1", "A metric");
  gauge.Increment("a");
  gauge.IncrementBy(2, "a");
  gauge.Set(3, "b");

  EXPECT_EQ(3, gauge.Get("a"));
  EXPECT_EQ(3, gauge.GetMax("b"));

  auto metric = gauge.Collect();
  std::sort(metric.gauges.begin(), metric.gauges.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_EQ("/tensorstore/gauge3", metric.metric_name);
  EXPECT_THAT(metric.field_names, ::testing::ElementsAre("field1"));
  ASSERT_EQ(2, metric.gauges.size());
  EXPECT_THAT(metric.gauges[0].fields, ::testing::ElementsAre("a"));
  EXPECT_EQ(3, std::get<int64_t>(metric.gauges[0].value));
  EXPECT_EQ(3, std::get<int64_t>(metric.gauges[0].max_value));
  EXPECT_THAT(metric.gauges[1].fields, ::testing::ElementsAre("b"));
  EXPECT_EQ(3, std::get<int64_t>(metric.gauges[1].value));
  EXPECT_EQ(3, std::get<int64_t>(metric.gauges[1].max_value));
}

TEST(MetricTest, GaugeDoubleFields) {
  auto& gauge =
      Gauge<double, bool>::New("/tensorstore/gauge4", "field1", "A metric");
  gauge.Increment(false);
  gauge.IncrementBy(2, false);
  gauge.Set(3, true);

  auto metric = gauge.Collect();
  std::sort(metric.gauges.begin(), metric.gauges.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_EQ("/tensorstore/gauge4", metric.metric_name);
  EXPECT_THAT(metric.field_names, ::testing::ElementsAre("field1"));
  ASSERT_EQ(2, metric.gauges.size());
  EXPECT_THAT(metric.gauges[0].fields, ::testing::ElementsAre("0"));
  EXPECT_EQ(3, std::get<double>(metric.gauges[0].value));
  EXPECT_EQ(3, std::get<double>(metric.gauges[0].max_value));
  EXPECT_THAT(metric.gauges[1].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(3, std::get<double>(metric.gauges[1].value));
  EXPECT_EQ(3, std::get<double>(metric.gauges[1].max_value));
}

TEST(MetricTest, Histogram) {
  auto& histogram =
      Histogram<DefaultBucketer>::New("/tensorstore/hist1", "A metric");
  histogram.Observe(1);
  histogram.Observe(2);
  histogram.Observe(1000);

  EXPECT_EQ(3, histogram.GetCount());
  EXPECT_EQ(1003, histogram.GetSum());

  auto metric = histogram.Collect();

  EXPECT_EQ("/tensorstore/hist1", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.histograms.size());
  EXPECT_TRUE(metric.histograms[0].fields.empty());
  EXPECT_EQ(1003, metric.histograms[0].sum);
  EXPECT_EQ(0, metric.histograms[0].buckets[0]);  // [-inf..0)
  EXPECT_EQ(0, metric.histograms[0].buckets[1]);  // <1
  EXPECT_EQ(1, metric.histograms[0].buckets[2]);  // <2
  EXPECT_EQ(1, metric.histograms[0].buckets[3]);  // <4
}

TEST(MetricTest, HistogramFields) {
  auto& histogram = Histogram<DefaultBucketer, int>::New("/tensorstore/hist2",
                                                         "field1", "A metric");
  histogram.Observe(-1.0, 1);  // =0
  histogram.Observe(0.11, 2);  // =1
  histogram.Observe(1.2, 3);   // =2
  histogram.Observe(2.1, 4);   // =3

  auto metric = histogram.Collect();
  EXPECT_EQ("/tensorstore/hist2", metric.metric_name);
  EXPECT_THAT(metric.field_names, ::testing::ElementsAre("field1"));

  std::sort(metric.histograms.begin(), metric.histograms.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });
  ASSERT_EQ(4, metric.histograms.size());
  EXPECT_THAT(metric.histograms[0].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(1, metric.histograms[0].buckets[0]);  // <0

  EXPECT_THAT(metric.histograms[1].fields, ::testing::ElementsAre("2"));
  EXPECT_EQ(1, metric.histograms[1].buckets[1]);  // <1

  EXPECT_THAT(metric.histograms[2].fields, ::testing::ElementsAre("3"));
  EXPECT_EQ(1, metric.histograms[2].buckets[2]);  // <2

  EXPECT_THAT(metric.histograms[3].fields, ::testing::ElementsAre("4"));
  EXPECT_EQ(1, metric.histograms[3].buckets[3]);  // <4
}

TEST(MetricTest, ValueInt) {
  auto& value = Value<int64_t>::New("/tensorstore/value1", "A metric");
  value.Set(3);
  EXPECT_EQ(3, value.Get());

  auto metric = value.Collect();

  EXPECT_EQ("/tensorstore/value1", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.values.size());
  EXPECT_TRUE(metric.values[0].fields.empty());

  EXPECT_TRUE(std::holds_alternative<int64_t>(metric.values[0].value));
  EXPECT_FALSE(std::holds_alternative<double>(metric.values[0].value));
  EXPECT_FALSE(std::holds_alternative<std::string>(metric.values[0].value));
  EXPECT_EQ(3, std::get<int64_t>(metric.values[0].value));
}

TEST(MetricTest, ValueString) {
  auto& gauge = Value<std::string>::New("/tensorstore/value2", "A metric");
  gauge.Set("foo");

  auto metric = gauge.Collect();

  EXPECT_EQ("/tensorstore/value2", metric.metric_name);
  EXPECT_TRUE(metric.field_names.empty());
  ASSERT_EQ(1, metric.values.size());
  EXPECT_TRUE(metric.values[0].fields.empty());

  EXPECT_FALSE(std::holds_alternative<int64_t>(metric.values[0].value));
  EXPECT_FALSE(std::holds_alternative<double>(metric.values[0].value));
  EXPECT_TRUE(std::holds_alternative<std::string>(metric.values[0].value));
  EXPECT_EQ("foo", std::get<std::string>(metric.values[0].value));
}

}  // namespace
