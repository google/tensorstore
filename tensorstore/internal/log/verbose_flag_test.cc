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

#include "tensorstore/internal/log/verbose_flag.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"

using ::tensorstore::internal_log::UpdateVerboseLogging;
using ::tensorstore::internal_log::VerboseFlag;

/// Ensure a VerboseFlag is properly initialized:
/// auto& verbose_logging = TENSORSTORE_LOG_FLAG("verbose");
#define TENSORSTORE_VERBOSE_FLAG(X)                                          \
  []() -> ::tensorstore::internal_log::VerboseFlag& {                        \
    ABSL_CONST_INIT static ::tensorstore::internal_log::VerboseFlag flag(X); \
    return flag;                                                             \
  }()

namespace {

TEST(VerboseFlag, Basic) {
  // environment is parsed on first variable access.
  UpdateVerboseLogging("a=2", true);

  ABSL_CONST_INIT static VerboseFlag a1("a");
  auto& b = TENSORSTORE_VERBOSE_FLAG("b");

  EXPECT_THAT((bool)a1, true);
  EXPECT_THAT(a1.Level(0), true);
  EXPECT_THAT(a1.Level(1), true);
  EXPECT_THAT(a1.Level(2), true);
  EXPECT_THAT(a1.Level(3), false);

  EXPECT_THAT((bool)b, false);
  EXPECT_THAT(b.Level(0), false);

  UpdateVerboseLogging("b,a=-1", false);

  EXPECT_THAT((bool)a1, false);
  EXPECT_THAT(a1.Level(0), false);
  EXPECT_THAT(a1.Level(1), false);

  EXPECT_THAT((bool)b, true);
  EXPECT_THAT(b.Level(0), true);
  EXPECT_THAT(b.Level(1), false);
}

}  // namespace
