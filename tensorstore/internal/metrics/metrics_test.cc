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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/domain_field.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/max_gauge.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/metrics/value.h"

namespace {

using ::tensorstore::internal_metrics::Counter;
using ::tensorstore::internal_metrics::DefaultBucketer;
using ::tensorstore::internal_metrics::DomainField;
using ::tensorstore::internal_metrics::Gauge;
using ::tensorstore::internal_metrics::GetMetricRegistry;
using ::tensorstore::internal_metrics::Histogram;
using ::tensorstore::internal_metrics::MaxGauge;
using ::tensorstore::internal_metrics::MetricMetadata;
using ::tensorstore::internal_metrics::Value;

TEST(MetricTest, CounterInt) {
  static Counter<int64_t> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/counter1", "A metric"));
    return true;
  }();

  counter.Increment();
  counter.IncrementBy(2);

  EXPECT_EQ(3, counter.Get());

  {
    auto metric = GetMetricRegistry().Collect("/tensorstore/counter1");
    ASSERT_TRUE(metric.has_value());

    EXPECT_EQ("/tensorstore/counter1", metric->metric_name);
    EXPECT_TRUE(metric->field_names.empty());
    ASSERT_EQ(1, metric->values.size());
    EXPECT_TRUE(metric->values[0].fields.empty());
    EXPECT_EQ(3, std::get<int64_t>(metric->values[0].value));
  }
}

TEST(MetricTest, CounterDouble) {
  static Counter<double> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/counter2", "A metric"));
    return true;
  }();

  counter.Increment();
  counter.IncrementBy(2);

  auto metric = GetMetricRegistry().Collect("/tensorstore/counter2");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/counter2", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());
  EXPECT_EQ(3, std::get<double>(metric->values[0].value));
}

TEST(MetricTest, CounterIntFields) {
  static Counter<int64_t, std::string> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter,
        MetricMetadata("/tensorstore/counter3", "A metric", {"field1"}));
    return true;
  }();

  counter.Increment("a");
  counter.IncrementBy(2, "b");

  EXPECT_EQ(1, counter.Get("a"));
  EXPECT_EQ(2, counter.Get("b"));

  auto metric = GetMetricRegistry().Collect("/tensorstore/counter3");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/counter3", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("field1"));

  ASSERT_EQ(2, metric->values.size());
  std::sort(metric->values.begin(), metric->values.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_THAT(metric->values[0].fields, ::testing::ElementsAre("a"));
  EXPECT_EQ(1, std::get<int64_t>(metric->values[0].value));
  EXPECT_THAT(metric->values[1].fields, ::testing::ElementsAre("b"));
  EXPECT_EQ(2, std::get<int64_t>(metric->values[1].value));
}

TEST(MetricTest, CounterDoubleFields) {
  static Counter<double, int> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter,
        MetricMetadata("/tensorstore/counter4", "A metric", {"field1"}));
    return true;
  }();

  counter.Increment(1);
  counter.IncrementBy(2, 2);

  auto metric = GetMetricRegistry().Collect("/tensorstore/counter4");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/counter4", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("field1"));

  ASSERT_EQ(2, metric->values.size());
  std::sort(metric->values.begin(), metric->values.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_THAT(metric->values[0].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(1, std::get<double>(metric->values[0].value));
  EXPECT_THAT(metric->values[1].fields, ::testing::ElementsAre("2"));
  EXPECT_EQ(2, std::get<double>(metric->values[1].value));
}

TEST(MetricTest, GaugeInt) {
  static Gauge<int64_t> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/gauge1", "A metric"));
    return true;
  }();

  gauge.Set(3);
  gauge.Increment();
  gauge.IncrementBy(2);

  EXPECT_EQ(6, gauge.Get());
  EXPECT_EQ(6, gauge.GetMax());

  auto metric = GetMetricRegistry().Collect("/tensorstore/gauge1");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/gauge1", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());
  EXPECT_EQ(6, std::get<int64_t>(metric->values[0].value));
}

TEST(MetricTest, GaugeDouble) {
  static Gauge<double> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/gauge2", "A metric"));
    return true;
  }();

  gauge.Set(3);
  gauge.Increment();
  gauge.IncrementBy(2);

  auto metric = GetMetricRegistry().Collect("/tensorstore/gauge2");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/gauge2", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());
  EXPECT_EQ(6, std::get<double>(metric->values[0].value));
}

TEST(MetricTest, GaugeIntFields) {
  static Gauge<int64_t, std::string> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/gauge3", "A metric", {"field1"}));
    return true;
  }();

  gauge.Increment("a");
  gauge.IncrementBy(2, "a");
  gauge.Set(3, "b");

  EXPECT_EQ(3, gauge.Get("a"));
  EXPECT_EQ(3, gauge.GetMax("b"));

  auto metric = GetMetricRegistry().Collect("/tensorstore/gauge3");
  ASSERT_TRUE(metric.has_value());
  std::sort(metric->values.begin(), metric->values.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_EQ("/tensorstore/gauge3", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("field1"));
  ASSERT_EQ(2, metric->values.size());
  EXPECT_THAT(metric->values[0].fields, ::testing::ElementsAre("a"));
  EXPECT_EQ(3, std::get<int64_t>(metric->values[0].value));
  EXPECT_EQ(3, std::get<int64_t>(metric->values[0].max_value));
  EXPECT_THAT(metric->values[1].fields, ::testing::ElementsAre("b"));
  EXPECT_EQ(3, std::get<int64_t>(metric->values[1].value));
  EXPECT_EQ(3, std::get<int64_t>(metric->values[1].max_value));
}

TEST(MetricTest, GaugeDoubleFields) {
  static Gauge<double, bool> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/gauge4", "A metric", {"field1"}));
    return true;
  }();

  gauge.Increment(false);
  gauge.IncrementBy(2, false);
  gauge.Set(3, true);

  auto metric = GetMetricRegistry().Collect("/tensorstore/gauge4");
  ASSERT_TRUE(metric.has_value());
  std::sort(metric->values.begin(), metric->values.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });

  EXPECT_EQ("/tensorstore/gauge4", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("field1"));
  ASSERT_EQ(2, metric->values.size());
  EXPECT_THAT(metric->values[0].fields, ::testing::ElementsAre("0"));
  EXPECT_EQ(3, std::get<double>(metric->values[0].value));
  EXPECT_EQ(3, std::get<double>(metric->values[0].max_value));
  EXPECT_THAT(metric->values[1].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(3, std::get<double>(metric->values[1].value));
  EXPECT_EQ(3, std::get<double>(metric->values[1].max_value));
}

TEST(MetricTest, MaxGauge) {
  static MaxGauge<double> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/max_gauge", "A metric"));
    return true;
  }();

  gauge.Set(3);
  gauge.Set(7);

  auto metric = GetMetricRegistry().Collect("/tensorstore/max_gauge");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/max_gauge", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());
  EXPECT_EQ(7, std::get<double>(metric->values[0].max_value));
}

TEST(MetricTest, DefaultBucketer) {
  // Boundary cases.
  EXPECT_EQ(DefaultBucketer::UnderflowBucket,
            DefaultBucketer::BucketForValue(-1));
  EXPECT_EQ(1, DefaultBucketer::BucketForValue(0));
  EXPECT_EQ(2, DefaultBucketer::BucketForValue(1));

  double v = std::nextafter(static_cast<double>(1ull << 63), 0);
  EXPECT_EQ(64, DefaultBucketer::BucketForValue(v));

  EXPECT_EQ(65,
            DefaultBucketer::BucketForValue(static_cast<double>(1ull << 63)));

  EXPECT_EQ(65, DefaultBucketer::BucketForValue(
                    std::numeric_limits<uint64_t>::max()));

  // Labels
  EXPECT_EQ("0",
            DefaultBucketer::LabelForBucket(DefaultBucketer::UnderflowBucket));
  EXPECT_EQ("1", DefaultBucketer::LabelForBucket(1));
  EXPECT_EQ("2", DefaultBucketer::LabelForBucket(2));
  EXPECT_EQ("4", DefaultBucketer::LabelForBucket(3));
  EXPECT_EQ("1M", DefaultBucketer::LabelForBucket(20));
  EXPECT_EQ("64E", DefaultBucketer::LabelForBucket(
                       DefaultBucketer::OverflowBucket - 3));
  EXPECT_EQ("128E", DefaultBucketer::LabelForBucket(
                        DefaultBucketer::OverflowBucket - 2));
  EXPECT_EQ("256E", DefaultBucketer::LabelForBucket(
                        DefaultBucketer::OverflowBucket - 1));

  EXPECT_EQ("Inf",
            DefaultBucketer::LabelForBucket(DefaultBucketer::OverflowBucket));
}

TEST(MetricTest, Histogram) {
  static Histogram<DefaultBucketer> histogram;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &histogram, MetricMetadata("/tensorstore/hist1", "A metric"));
    return true;
  }();

  histogram.Observe(1);
  histogram.Observe(2);
  histogram.Observe(1000);

  EXPECT_EQ(3, histogram.GetCount());
  EXPECT_NEAR(334.333333, histogram.GetMean(), 0.001);

  auto metric = GetMetricRegistry().Collect("/tensorstore/hist1");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/hist1", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->histograms.size());
  EXPECT_TRUE(metric->histograms[0].fields.empty());
  EXPECT_NEAR(334.333333, metric->histograms[0].mean, 0.001);
  EXPECT_EQ(0, metric->histograms[0].buckets[0]);  // [-inf..0)
  EXPECT_EQ(0, metric->histograms[0].buckets[1]);  // <1
  EXPECT_EQ(1, metric->histograms[0].buckets[2]);  // <2
  EXPECT_EQ(1, metric->histograms[0].buckets[3]);  // <4
}

TEST(MetricTest, HistogramFields) {
  static Histogram<DefaultBucketer, int> histogram;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &histogram,
        MetricMetadata("/tensorstore/hist2", "A metric", {"field1"}));
    return true;
  }();

  histogram.Observe(-1.0, 1);  // =0
  histogram.Observe(0.11, 2);  // =1
  histogram.Observe(1.2, 3);   // =2
  histogram.Observe(2.1, 4);   // =3

  auto metric = GetMetricRegistry().Collect("/tensorstore/hist2");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/hist2", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("field1"));

  std::sort(metric->histograms.begin(), metric->histograms.end(),
            [](auto& a, auto& b) { return a.fields < b.fields; });
  ASSERT_EQ(4, metric->histograms.size());
  EXPECT_THAT(metric->histograms[0].fields, ::testing::ElementsAre("1"));
  EXPECT_EQ(1, metric->histograms[0].buckets[0]);  // <0

  EXPECT_THAT(metric->histograms[1].fields, ::testing::ElementsAre("2"));
  EXPECT_EQ(1, metric->histograms[1].buckets[1]);  // <1

  EXPECT_THAT(metric->histograms[2].fields, ::testing::ElementsAre("3"));
  EXPECT_EQ(1, metric->histograms[2].buckets[2]);  // <2

  EXPECT_THAT(metric->histograms[3].fields, ::testing::ElementsAre("4"));
  EXPECT_EQ(1, metric->histograms[3].buckets[3]);  // <4
}

TEST(MetricTest, ValueInt) {
  static Value<int64_t> value;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &value, MetricMetadata("/tensorstore/value1", "A metric"));
    return true;
  }();

  value.Set(3);
  EXPECT_EQ(3, value.Get());

  auto metric = GetMetricRegistry().Collect("/tensorstore/value1");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/value1", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());

  EXPECT_TRUE(std::holds_alternative<int64_t>(metric->values[0].value));
  EXPECT_FALSE(std::holds_alternative<double>(metric->values[0].value));
  EXPECT_FALSE(std::holds_alternative<std::string>(metric->values[0].value));
  EXPECT_EQ(3, std::get<int64_t>(metric->values[0].value));
}

TEST(MetricTest, ValueString) {
  static Value<std::string> gauge;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &gauge, MetricMetadata("/tensorstore/value2", "A metric"));
    return true;
  }();

  gauge.Set("foo");

  auto metric = GetMetricRegistry().Collect("/tensorstore/value2");
  ASSERT_TRUE(metric.has_value());

  EXPECT_EQ("/tensorstore/value2", metric->metric_name);
  EXPECT_TRUE(metric->field_names.empty());
  ASSERT_EQ(1, metric->values.size());
  EXPECT_TRUE(metric->values[0].fields.empty());

  EXPECT_FALSE(std::holds_alternative<int64_t>(metric->values[0].value));
  EXPECT_FALSE(std::holds_alternative<double>(metric->values[0].value));
  EXPECT_TRUE(std::holds_alternative<std::string>(metric->values[0].value));
  EXPECT_EQ("foo", std::get<std::string>(metric->values[0].value));
}

TEST(MetricTest, IdempotentRegistration) {
  static Counter<int64_t> counter;
  GetMetricRegistry().Register(
      &counter, MetricMetadata("/tensorstore/idempotent_test", "First"));
  GetMetricRegistry().Register(
      &counter, MetricMetadata("/tensorstore/idempotent_test", "Second"));

  counter.Increment();
  EXPECT_EQ(1, counter.Get());
}

TEST(MetricTest, ConflictingRegistrationDeathTest) {
  static Counter<int64_t> counter1;
  static Counter<int64_t> counter2;
  GetMetricRegistry().Register(
      &counter1, MetricMetadata("/tensorstore/conflict_test", "First"));
  EXPECT_DEATH(
      GetMetricRegistry().Register(
          &counter2, MetricMetadata("/tensorstore/conflict_test", "Second")),
      "Metric path conflict");
}

TEST(MetricTest, CounterFiveFields) {
  static Counter<int64_t, std::string, std::string, std::string, std::string,
                 std::string>
      counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter,
        MetricMetadata("/tensorstore/counter_5_fields", "A 5-field metric",
                       {"f1", "f2", "f3", "f4", "f5"}));
    return true;
  }();

  counter.Increment("a", "b", "c", "d", "e");
  EXPECT_EQ(1, counter.Get("a", "b", "c", "d", "e"));

  auto metric = GetMetricRegistry().Collect("/tensorstore/counter_5_fields");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/counter_5_fields", metric->metric_name);
  EXPECT_THAT(metric->field_names,
              ::testing::ElementsAre("f1", "f2", "f3", "f4", "f5"));
  ASSERT_EQ(1, metric->values.size());
  EXPECT_THAT(metric->values[0].fields,
              ::testing::ElementsAre("a", "b", "c", "d", "e"));
  EXPECT_EQ(1, std::get<int64_t>(metric->values[0].value));
}

// This should fail to compile if uncommented, because MetricMetadata
// constructor uses StaticStringView which deletes std::string overloads.
// TEST(MetricTest, InvalidStringMetadata) {
//   std::string dynamic_name = "/tensorstore/dynamic";
//   MetricMetadata metadata(dynamic_name, "desc");
// }

struct TestMethodDomain {
  static constexpr std::array<std::string_view, 3> kValues = {"Read", "Write",
                                                              "Delete"};
  // FIND_SEED Read Write Delete
  static constexpr uint32_t kSeed = 0;
  static constexpr size_t kTableSize = 3;
};

struct TestStatusDomain {
  static constexpr std::array<std::string_view, 2> kValues = {"Ok", "Error"};
  // FIND_SEED Ok Error
  static constexpr uint32_t kSeed = 2;
  static constexpr size_t kTableSize = 2;
};

using MethodField =
    tensorstore::internal_metrics::DomainField<TestMethodDomain>;
using StatusField =
    tensorstore::internal_metrics::DomainField<TestStatusDomain>;

TEST(MetricTest, PerfectHashCounter) {
  static Counter<int64_t, MethodField, StatusField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/perfect_hash_counter",
                                 "A metric", {"method", "status"}));
    return true;
  }();

  // Test valid labels (case-insensitive)
  counter.Increment("read", "ok");
  counter.IncrementBy(2, "Read", "OK");
  counter.Increment("WRITE", "error");

  // Test invalid labels (should go to overflow cell)
  counter.Increment("Delete", "InvalidStatus");
  counter.Increment("InvalidMethod", "Ok");
  counter.Increment("InvalidMethod", "InvalidStatus");

  EXPECT_EQ(3, counter.Get("Read", "Ok"));
  EXPECT_EQ(3, counter.Get("read", "ok"));  // Test case-insensitive Get
  EXPECT_EQ(1, counter.Get("Write", "Error"));
  EXPECT_EQ(1, counter.Get("WRITE", "error"));  // Test case-insensitive Get
  EXPECT_EQ(0, counter.Get("Delete", "Ok"));

  auto metric =
      GetMetricRegistry().Collect("/tensorstore/perfect_hash_counter");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/perfect_hash_counter", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("method", "status"));

  int64_t non_zero_count = 0;
  int64_t overflow_count = 0;
  for (const auto& val : metric->values) {
    int64_t v = std::get<int64_t>(val.value);
    if (v > 0) {
      non_zero_count++;
      if (val.fields[0].empty() && val.fields[1].empty()) {
        overflow_count = v;
      }
    }
  }

  EXPECT_EQ(3, non_zero_count);
  EXPECT_EQ(3, overflow_count);
}

enum class TestTask {
  kTaskA = 10,
  kTaskB = 11,
  kTaskC = 12,
};

struct TestTaskDomain {
  static constexpr std::array<TestTask, 3> kValues = {
      TestTask::kTaskA, TestTask::kTaskB, TestTask::kTaskC};
};

using TaskField = DomainField<TestTaskDomain>;

TEST(MetricTest, DenseDomainCounter) {
  static Counter<int64_t, TaskField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/dense_domain_counter",
                                 "A metric", {"task"}));
    return true;
  }();

  // Test valid labels
  counter.Increment(TestTask::kTaskA);
  counter.IncrementBy(2, TestTask::kTaskB);
  counter.Increment(TestTask::kTaskC);

  // Test invalid label (should go to overflow cell)
  counter.Increment(static_cast<TestTask>(9));
  counter.Increment(static_cast<TestTask>(13));

  EXPECT_EQ(1, counter.Get(TestTask::kTaskA));
  EXPECT_EQ(2, counter.Get(TestTask::kTaskB));
  EXPECT_EQ(1, counter.Get(TestTask::kTaskC));
  EXPECT_EQ(0, counter.Get(static_cast<TestTask>(9)));

  auto metric =
      GetMetricRegistry().Collect("/tensorstore/dense_domain_counter");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/dense_domain_counter", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("task"));

  int64_t non_zero_count = 0;
  int64_t overflow_count = 0;
  for (const auto& val : metric->values) {
    int64_t v = std::get<int64_t>(val.value);
    if (v > 0) {
      non_zero_count++;
      if (val.fields[0].empty()) {
        overflow_count = v;
      }
    }
  }

  // We have 3 valid cells + 1 overflow cell that are non-zero.
  EXPECT_EQ(4, non_zero_count);
  EXPECT_EQ(2, overflow_count);
}

TEST(MetricTest, MixedDomainCounter) {
  static Counter<int64_t, MethodField, TaskField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/mixed_domain_counter",
                                 "A mixed domain metric", {"method", "task"}));
    return true;
  }();

  // Test valid labels
  counter.Increment("read", TestTask::kTaskA);
  counter.IncrementBy(2, "write", TestTask::kTaskB);

  // Test invalid label mapping to overflow
  counter.Increment("invalid_method", TestTask::kTaskA);
  counter.Increment("read", static_cast<TestTask>(9));

  EXPECT_EQ(1, counter.Get("read", TestTask::kTaskA));
  EXPECT_EQ(2, counter.Get("write", TestTask::kTaskB));
  EXPECT_EQ(0, counter.Get("read", static_cast<TestTask>(9)));

  auto metric =
      GetMetricRegistry().Collect("/tensorstore/mixed_domain_counter");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/mixed_domain_counter", metric->metric_name);
  EXPECT_THAT(metric->field_names, ::testing::ElementsAre("method", "task"));

  int64_t non_zero_count = 0;
  int64_t overflow_count = 0;
  for (const auto& val : metric->values) {
    int64_t v = std::get<int64_t>(val.value);
    if (v > 0) {
      non_zero_count++;
      if (val.fields[0].empty() && val.fields[1].empty()) {
        overflow_count = v;
      }
    }
  }

  EXPECT_EQ(3, non_zero_count);  // 2 valid cells + 1 overflow cell
  EXPECT_EQ(2, overflow_count);  // 2 invalid increments mapped to overflow
}

TEST(MetricTest, MixedDomainBoolCounter) {
  static Counter<int64_t, bool, MethodField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter,
        MetricMetadata("/tensorstore/mixed_domain_bool_counter",
                       "A mixed domain bool metric", {"is_active", "method"}));
    return true;
  }();

  // Test increments
  counter.Increment(true, "read");
  counter.IncrementBy(5, false, "write");

  // Test invalid label mapping to overflow
  counter.Increment(true, "invalid_method");

  EXPECT_EQ(1, counter.Get(true, "read"));
  EXPECT_EQ(5, counter.Get(false, "write"));
  EXPECT_EQ(0, counter.Get(true, "invalid_method"));

  auto metric =
      GetMetricRegistry().Collect("/tensorstore/mixed_domain_bool_counter");
  ASSERT_TRUE(metric.has_value());
  EXPECT_EQ("/tensorstore/mixed_domain_bool_counter", metric->metric_name);
  EXPECT_THAT(metric->field_names,
              ::testing::ElementsAre("is_active", "method"));

  int64_t non_zero_count = 0;
  int64_t overflow_count = 0;
  for (const auto& val : metric->values) {
    int64_t v = std::get<int64_t>(val.value);
    if (v > 0) {
      non_zero_count++;
      if (val.fields[0] == "0" && val.fields[1].empty()) {
        overflow_count = v;
      }
    }
  }

  EXPECT_EQ(3, non_zero_count);  // 2 valid cells + 1 overflow cell
  EXPECT_EQ(1, overflow_count);  // 1 invalid increment mapped to overflow
}

TEST(MetricTest, PerfectHashMetricImplEndToEnd) {
  static Counter<int64_t, MethodField> counter;
  static const bool registered [[maybe_unused]] = [&] {
    GetMetricRegistry().Register(
        &counter, MetricMetadata("/tensorstore/perfect_hash_e2e", "A metric",
                                 {"method"}));
    return true;
  }();

  // 1. Valid domain values -> correct cell
  counter.Increment("Read");
  counter.IncrementBy(3, "Write");
  EXPECT_EQ(1, counter.Get("Read"));
  EXPECT_EQ(3, counter.Get("Write"));

  // 2. Invalid domain values -> overflow cell
  counter.Increment("InvalidValue");
  EXPECT_EQ(0, counter.Get("InvalidValue"));

  // 3. CollectCells -> emits correct labels
  auto metric = GetMetricRegistry().Collect("/tensorstore/perfect_hash_e2e");
  ASSERT_TRUE(metric.has_value());

  int64_t read_val = 0;
  int64_t write_val = 0;
  int64_t overflow_val = 0;

  for (const auto& val : metric->values) {
    int64_t v = std::get<int64_t>(val.value);
    if (val.fields[0] == "Read") {
      read_val = v;
    } else if (val.fields[0] == "Write") {
      write_val = v;
    } else if (val.fields[0].empty()) {
      overflow_val = v;
    }
  }

  EXPECT_EQ(1, read_val);
  EXPECT_EQ(3, write_val);
  EXPECT_EQ(1, overflow_val);

  // 4. Reset -> clears all cells including overflow
  GetMetricRegistry().Reset();
  EXPECT_EQ(0, counter.Get("Read"));
  EXPECT_EQ(0, counter.Get("Write"));

  auto metric_after_reset =
      GetMetricRegistry().Collect("/tensorstore/perfect_hash_e2e");
  ASSERT_TRUE(metric_after_reset.has_value());
  for (const auto& val : metric_after_reset->values) {
    EXPECT_EQ(0, std::get<int64_t>(val.value));
  }
}

}  // namespace

#endif  // !defined(TENSORSTORE_METRICS_DISABLED)
