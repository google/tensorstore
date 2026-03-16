// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/downsample_method.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::DownsampleMethod;

TEST(DownsampleMethodTest, AbslStringify) {
  EXPECT_EQ("stride", absl::StrCat(DownsampleMethod::kStride));
  EXPECT_EQ("mean", absl::StrCat(DownsampleMethod::kMean));
  EXPECT_EQ("median", absl::StrCat(DownsampleMethod::kMedian));
  EXPECT_EQ("mode", absl::StrCat(DownsampleMethod::kMode));
  EXPECT_EQ("min", absl::StrCat(DownsampleMethod::kMin));
  EXPECT_EQ("max", absl::StrCat(DownsampleMethod::kMax));
}

}  // namespace
