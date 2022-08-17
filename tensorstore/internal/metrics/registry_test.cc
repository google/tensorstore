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

#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/internal/metrics/collect.h"

namespace {

using ::tensorstore::internal_metrics::CollectedMetric;
using ::tensorstore::internal_metrics::MetricRegistry;

TEST(RegistryTest, Arbitrary) {
  MetricRegistry registry;

  registry.AddGeneric("/my/metric", [] {
    CollectedMetric metric;
    metric.metric_name = "/my/metric";
    return metric;
  });

  registry.AddGeneric("/my/metric2", [] {
    CollectedMetric metric;
    metric.metric_name = "/my/metric2";
    return metric;
  });

  EXPECT_FALSE(registry.Collect("/my/foo").has_value());

  auto collected = registry.Collect("/my/metric");
  ASSERT_TRUE(collected.has_value());
  EXPECT_EQ("/my/metric", collected->metric_name);

  auto all = registry.CollectWithPrefix("/my");
  EXPECT_EQ(2, all.size());
}

}  // namespace
