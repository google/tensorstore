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

#include "tensorstore/internal/source_location.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace {

using ::tensorstore::SourceLocation;

uint64_t TakesSourceLocation(
    SourceLocation loc = tensorstore::SourceLocation::current()) {
  return loc.line();
}

TEST(SourceLocationTest, Basic) {
  constexpr tensorstore::SourceLocation loc =
      tensorstore::SourceLocation::current();
  EXPECT_NE(0, loc.line());
  EXPECT_NE(1, loc.line());

  EXPECT_NE(1, TakesSourceLocation(tensorstore::SourceLocation::current()));
}

}  // namespace
