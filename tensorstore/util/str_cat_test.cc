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

#include "tensorstore/util/str_cat.h"

#include <complex>
#include <ostream>
#include <string>

#include <gtest/gtest.h>

namespace {

struct X {
  ~X() {}
  int value;
};

enum class OstreamableEnum { value = 0 };
enum class PlainEnum { value = 0 };

std::ostream& operator<<(std::ostream& os, OstreamableEnum e) {
  return os << "enum";
}

TEST(ToStringUsingOstreamTest, Basic) {
  EXPECT_EQ("hello", tensorstore::ToStringUsingOstream("hello"));
  EXPECT_EQ("1", tensorstore::ToStringUsingOstream(1));
  EXPECT_EQ("(1,2)",
            tensorstore::ToStringUsingOstream(std::complex<float>(1, 2)));
}

TEST(StrAppendTest, Basic) {
  std::string result = "X";
  tensorstore::StrAppend(&result, "a", std::complex<float>(1, 2), 3);
  EXPECT_EQ("Xa(1,2)3", result);
}

TEST(StrCat, Basic) {
  EXPECT_EQ("a(1,2)3", tensorstore::StrCat("a", std::complex<float>(1, 2), 3));
}

TEST(StrCat, Enum) {
  EXPECT_EQ("enum", tensorstore::StrCat(OstreamableEnum::value));
  EXPECT_EQ("0", tensorstore::StrCat(PlainEnum::value));
}

TEST(StrCat, Unprintable) {
  EXPECT_EQ("<unprintable>", tensorstore::StrCat(X{5}));
}

TEST(SpanTest, Ostream) {
  std::ostringstream ostr;
  ostr << tensorstore::span({1, 2, 3});
  EXPECT_EQ("{1, 2, 3}", ostr.str());
}

}  // namespace
