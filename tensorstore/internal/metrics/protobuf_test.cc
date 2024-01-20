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

#ifndef TENSORSTORE_METRICS_DISABLED

#include "tensorstore/internal/metrics/protobuf.h"

#include <stdint.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metrics.pb.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/proto/protobuf_matchers.h"

namespace {

using ::protobuf_matchers::Approximately;
using ::protobuf_matchers::EqualsProto;
using ::protobuf_matchers::IgnoringRepeatedFieldOrdering;
using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::Counter;
using ::tensorstore::internal_metrics::DefaultBucketer;
using ::tensorstore::internal_metrics::Gauge;
using ::tensorstore::internal_metrics::GetMetricRegistry;
using ::tensorstore::internal_metrics::Histogram;
using ::tensorstore::internal_metrics::Value;

TEST(ProtobufTest, BasicConversion) {
  CollectedMetric metric;
  metric.metric_name = "abc";
  metric.tag = "tag";

  /// NOTE: A single metric must not mix counters...

  // Gauges.
  metric.values.emplace_back(
      CollectedMetric::Value{{"c", "d"}, int64_t{1}, int64_t{2}});
  metric.values.emplace_back(CollectedMetric::Value{{"e", "g"}, 2.3, 3.4});

  // Counters.
  metric.values.emplace_back(CollectedMetric::Value{{}, int64_t{1}});
  metric.values.emplace_back(CollectedMetric::Value{{"i"}, 1.2});
  metric.values.emplace_back(CollectedMetric::Value{{}, "boo"});

  // Histograms.
  metric.histograms.emplace_back(CollectedMetric::Histogram{
      {"h"}, /*count*/ 10, /*mean*/ 1, /*ssd*/ 1, {1, 1, 1, 1, 1}});

  tensorstore::metrics_proto::Metric proto;
  tensorstore::internal_metrics::CollectedMetricToProto(metric, proto);

  EXPECT_THAT(proto,
              IgnoringRepeatedFieldOrdering(Approximately(EqualsProto(R"pb(
                metric_name: "abc"
                tag: "tag"
                metadata {}
                instance {
                  field: "c"
                  field: "d"
                  int_value { value: 1 max_value: 2 }
                }
                instance {
                  field: "e"
                  field: "g"
                  double_value { value: 2.3 max_value: 3.4 }
                }
                instance { int_value { value: 1 } }
                instance {
                  field: "i"
                  double_value { value: 1.2 }
                }
                instance { string_value { value: "boo" } }
                instance {
                  field: "h"
                  histogram {
                    count: 10
                    mean: 1
                    sum_of_squared_deviation: 1
                    bucket: 1
                    bucket: 1
                    bucket: 1
                    bucket: 1
                    bucket: 1
                  }
                }
              )pb"))));
}

TEST(ProtobufTest, FromRegistry) {
  {
    auto& counter =
        Counter<int64_t>::New("/protobuf_test/counter1", "A metric");
    counter.Increment();
    counter.IncrementBy(2);
  }
  {
    auto& counter = Counter<double>::New("/protobuf_test/counter2", "A metric");
    counter.Increment();
    counter.IncrementBy(2);
  }
  {
    auto& counter = Counter<int64_t, std::string>::New(
        "/protobuf_test/counter3", "field1", "A metric");
    counter.Increment("a");
    counter.IncrementBy(2, "b");
  }
  {
    auto& counter = Counter<double, int>::New("/protobuf_test/counter4",
                                              "field1", "A metric");
    counter.Increment(1);
    counter.IncrementBy(2, 2);
  }
  {
    auto& gauge = Gauge<int64_t>::New("/protobuf_test/gauge1", "A metric");
    gauge.Set(3);
    gauge.Increment();
    gauge.IncrementBy(2);
  }
  {
    auto& gauge = Gauge<double>::New("/protobuf_test/gauge2", "A metric");
    gauge.Set(3);
    gauge.Increment();
    gauge.IncrementBy(2);
  }
  {
    auto& gauge = Gauge<int64_t, std::string>::New("/protobuf_test/gauge3",
                                                   "field1", "A metric");
    gauge.Increment("a");
    gauge.IncrementBy(2, "a");
    gauge.Set(3, "b");
  }
  {
    auto& gauge =
        Gauge<double, bool>::New("/protobuf_test/gauge4", "field1", "A metric");
    gauge.Increment(false);
    gauge.IncrementBy(2, false);
    gauge.Set(3, true);
  }

  {
    auto& histogram =
        Histogram<DefaultBucketer>::New("/protobuf_test/hist1", "A metric");
    histogram.Observe(1);
    histogram.Observe(2);
    histogram.Observe(1000);
  }
  {
    auto& histogram = Histogram<DefaultBucketer, int>::New(
        "/protobuf_test/hist2", "field1", "A metric");
    histogram.Observe(-1.0, 1);  // =0
    histogram.Observe(0.11, 2);  // =1
    histogram.Observe(1.2, 3);   // =2
    histogram.Observe(2.1, 4);   // =3
  }
  {
    auto& value = Value<int64_t>::New("/protobuf_test/value1", "A metric");
    value.Set(3);
  }
  {
    auto& gauge = Value<std::string>::New("/protobuf_test/value2", "A metric");
    gauge.Set("foo");
  }

  tensorstore::metrics_proto::MetricCollection metric;
  tensorstore::internal_metrics::CollectedMetricToProtoCollection(
      GetMetricRegistry().CollectWithPrefix("/protobuf_test"), metric);

  tensorstore::internal_metrics::SortProtoCollection(metric);

  EXPECT_THAT(metric, Approximately(EqualsProto(R"pb(
                metric {
                  metric_name: "/protobuf_test/counter1"
                  tag: "counter"
                  metadata { description: "A metric" }
                  instance { int_value { value: 3 } }
                }
                metric {
                  metric_name: "/protobuf_test/counter2"
                  tag: "counter"
                  metadata { description: "A metric" }
                  instance { double_value { value: 3 } }
                }
                metric {
                  metric_name: "/protobuf_test/counter3"
                  tag: "counter"
                  field_name: "field1"
                  metadata { description: "A metric" }
                  instance {
                    field: "a"
                    int_value { value: 1 }
                  }
                  instance {
                    field: "b"
                    int_value { value: 2 }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/counter4"
                  tag: "counter"
                  field_name: "field1"
                  metadata { description: "A metric" }
                  instance {
                    field: "1"
                    double_value { value: 1 }
                  }
                  instance {
                    field: "2"
                    double_value { value: 2 }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/gauge1"
                  tag: "gauge"
                  metadata { description: "A metric" }
                  instance { int_value { value: 6 max_value: 6 } }
                }
                metric {
                  metric_name: "/protobuf_test/gauge2"
                  tag: "gauge"
                  metadata { description: "A metric" }
                  instance { double_value { value: 6 max_value: 6 } }
                }
                metric {
                  metric_name: "/protobuf_test/gauge3"
                  tag: "gauge"
                  field_name: "field1"
                  metadata { description: "A metric" }
                  instance {
                    field: "a"
                    int_value { value: 3 max_value: 3 }
                  }
                  instance {
                    field: "b"
                    int_value { value: 3 max_value: 3 }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/gauge4"
                  tag: "gauge"
                  field_name: "field1"
                  metadata { description: "A metric" }
                  instance {
                    field: "0"
                    double_value { value: 3 max_value: 3 }
                  }
                  instance {
                    field: "1"
                    double_value { value: 3 max_value: 3 }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/hist1"
                  tag: "default_histogram"
                  metadata { description: "A metric" }
                  instance {
                    histogram {
                      count: 3
                      mean: 334.33333333333331
                      sum_of_squared_deviation: 664668.66666666674
                      bucket: -2
                      bucket: 1
                      bucket: 1
                      bucket: -7
                      bucket: 1
                    }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/hist2"
                  tag: "default_histogram"
                  field_name: "field1"
                  metadata { description: "A metric" }
                  instance {
                    field: "1"
                    histogram { count: 1 mean: -1 bucket: 1 }
                  }
                  instance {
                    field: "2"
                    histogram { count: 1 mean: 0.11 bucket: -1 bucket: 1 }
                  }
                  instance {
                    field: "3"
                    histogram { count: 1 mean: 1.2 bucket: -2 bucket: 1 }
                  }
                  instance {
                    field: "4"
                    histogram { count: 1 mean: 2.1 bucket: -3 bucket: 1 }
                  }
                }
                metric {
                  metric_name: "/protobuf_test/value1"
                  tag: "value"
                  metadata { description: "A metric" }
                  instance { int_value { value: 3 } }
                }
                metric {
                  metric_name: "/protobuf_test/value2"
                  tag: "value"
                  metadata { description: "A metric" }
                  instance { string_value { value: "foo" } }
                }
              )pb")));
}

}  // namespace

#endif  // !defined(TENSORSTORE_METRICS_DISABLED)
