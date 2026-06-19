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

#include "tensorstore/internal/metrics/registry.h"
#ifndef TENSORSTORE_METRICS_DISABLED

#include <stdint.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/metrics.pb.h"
#include "tensorstore/internal/metrics/protobuf.h"
#include "tensorstore/internal/metrics/registration.h"
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
using ::tensorstore::internal_metrics::MetricMetadata;
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

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_counter1, Counter<int64_t>,
    MetricMetadata("/protobuf_test/counter1", "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_counter2, Counter<double>,
    MetricMetadata("/protobuf_test/counter2", "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_counter3, (Counter<int64_t, std::string>),
    MetricMetadata("/protobuf_test/counter3", "A metric"), "field1");

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_counter4, (Counter<double, int>),
    MetricMetadata("/protobuf_test/counter4", "A metric"), "field1");

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_gauge1, Gauge<int64_t>,
                                        MetricMetadata("/protobuf_test/gauge1",
                                                       "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_gauge2, Gauge<double>,
                                        MetricMetadata("/protobuf_test/gauge2",
                                                       "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_gauge3, (Gauge<int64_t, std::string>),
    MetricMetadata("/protobuf_test/gauge3", "A metric"), "field1");

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_gauge4, (Gauge<double, bool>),
                                        MetricMetadata("/protobuf_test/gauge4",
                                                       "A metric"),
                                        "field1");

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_hist1, Histogram<DefaultBucketer>,
                                        MetricMetadata("/protobuf_test/hist1",
                                                       "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(
    test_hist2, (Histogram<DefaultBucketer, int>),
    MetricMetadata("/protobuf_test/hist2", "A metric"), "field1");

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_value1, Value<int64_t>,
                                        MetricMetadata("/protobuf_test/value1",
                                                       "A metric"));

TENSORSTORE_DECLARE_AND_REGISTER_METRIC(test_value2, Value<std::string>,
                                        MetricMetadata("/protobuf_test/value2",
                                                       "A metric"));

TEST(ProtobufTest, FromRegistry) {
  test_counter1.Increment();
  test_counter1.IncrementBy(2);

  test_counter2.Increment();
  test_counter2.IncrementBy(2);

  test_counter3.Increment("a");
  test_counter3.IncrementBy(2, "b");

  test_counter4.Increment(1);
  test_counter4.IncrementBy(2, 2);

  test_gauge1.Set(3);
  test_gauge1.Increment();
  test_gauge1.IncrementBy(2);

  test_gauge2.Set(3);
  test_gauge2.Increment();
  test_gauge2.IncrementBy(2);

  test_gauge3.Increment("a");
  test_gauge3.IncrementBy(2, "a");
  test_gauge3.Set(3, "b");

  test_gauge4.Increment(false);
  test_gauge4.IncrementBy(2, false);
  test_gauge4.Set(3, true);

  test_hist1.Observe(1);
  test_hist1.Observe(2);
  test_hist1.Observe(1000);

  test_hist2.Observe(-1.0, 1);  // =0
  test_hist2.Observe(0.11, 2);  // =1
  test_hist2.Observe(1.2, 3);   // =2
  test_hist2.Observe(2.1, 4);   // =3

  test_value1.Set(3);

  test_value2.Set("foo");

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
