// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/util/quote_string.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::QuoteString;
using ::testing::StrEq;

TEST(QuoteStringTest, Basic) {
  EXPECT_THAT(QuoteString("abc "), StrEq("\"abc \""));
  EXPECT_THAT(QuoteString("a\"b\n\x01"), StrEq("\"a\\\"b\\n\\x01\""));
  EXPECT_THAT(QuoteString("'"), StrEq("\"\\'\""));
}

}  // namespace
