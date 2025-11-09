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

#include "tensorstore/progress.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::CopyProgress;
using ::tensorstore::ReadProgress;
using ::tensorstore::WriteProgress;

TEST(ReadProgressTest, Comparison) {
  ReadProgress a{1, 1};
  ReadProgress b{2, 2};
  ReadProgress c{2, 1};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(b, c);
}

TEST(ReadProgressTest, Ostream) {
  EXPECT_EQ("{ total_elements=2, copied_elements=1 }",
            tensorstore::StrCat(ReadProgress{2, 1}));
}

TEST(WriteProgressTest, Comparison) {
  WriteProgress a{1, 1, 1};
  WriteProgress b{2, 2, 2};
  WriteProgress c{2, 1, 1};
  WriteProgress d{2, 1, 2};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(b, d);
  EXPECT_NE(b, c);
  EXPECT_NE(c, d);
}

TEST(WriteProgressTest, Ostream) {
  EXPECT_EQ("{ total_elements=3, copied_elements=2, committed_elements=1 }",
            tensorstore::StrCat(WriteProgress{3, 2, 1}));
}

TEST(CopyProgressTest, Comparison) {
  CopyProgress a{1, 1, 1, 1};
  CopyProgress b{2, 1, 1, 1};
  CopyProgress c{1, 2, 1, 1};
  CopyProgress d{1, 1, 2, 1};
  CopyProgress e{1, 1, 1, 2};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_EQ(e, e);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(a, e);
}

TEST(CopyProgressTest, Ostream) {
  EXPECT_EQ(
      "{ total_elements=4, read_elements=3, copied_elements=2, "
      "committed_elements=1 }",
      tensorstore::StrCat(CopyProgress{4, 3, 2, 1}));
}

}  // namespace
