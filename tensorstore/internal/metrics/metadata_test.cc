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

#include "tensorstore/internal/metrics/metadata.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_metrics::IsValidMetricLabel;
using ::tensorstore::internal_metrics::IsValidMetricName;

TEST(MetadataTest, IsValidMetricName) {
  EXPECT_FALSE(IsValidMetricName(""));
  EXPECT_FALSE(IsValidMetricName("/"));
  EXPECT_FALSE(IsValidMetricName("//"));
  EXPECT_FALSE(IsValidMetricName("/foo/"));
  EXPECT_FALSE(IsValidMetricName("/foo//bar"));
  EXPECT_FALSE(IsValidMetricName("/_foo"));
  EXPECT_FALSE(IsValidMetricName("/foo%"));
  EXPECT_FALSE(IsValidMetricName("/foo%"));
  EXPECT_FALSE(IsValidMetricName("/foo.bar"));
  EXPECT_FALSE(IsValidMetricName("foo_1"));

  EXPECT_TRUE(IsValidMetricName("/foo/1_bar/Baz"));
}

TEST(MetadataTest, IsValidMetricLabel) {
  EXPECT_FALSE(IsValidMetricLabel(""));
  EXPECT_FALSE(IsValidMetricLabel("/"));
  EXPECT_FALSE(IsValidMetricLabel("1_bar"));
  EXPECT_FALSE(IsValidMetricLabel("_bar"));
  EXPECT_FALSE(IsValidMetricLabel("foo/bar"));
  EXPECT_FALSE(IsValidMetricLabel("foo-bar"));
  EXPECT_FALSE(IsValidMetricLabel("foo.bar"));

  EXPECT_TRUE(IsValidMetricLabel("a"));
  EXPECT_TRUE(IsValidMetricLabel("foB_1"));
}

}  // namespace
