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
#include <map>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::internal_strcat::StringifyUsingOstream;

enum class OstreamableEnum { value = 0 };
enum class PlainEnum { value = 0 };

std::ostream& operator<<(std::ostream& os, OstreamableEnum e) {
  return os << "enum";
}

TEST(ToStringUsingOstreamTest, Basic) {
  EXPECT_EQ("hello", StringifyUsingOstream("hello"));
  EXPECT_EQ("1", StringifyUsingOstream(1));
  EXPECT_EQ("(1,2)", StringifyUsingOstream(std::complex<float>(1, 2)));
}

TEST(StrAppendTest, Basic) {
  std::string result = "X";
  tensorstore::StrAppend(&result, "a", std::complex<float>(1, 2), 3);
  EXPECT_EQ("Xa(1,2)3", result);
}

TEST(StrCat, Basic) {
  EXPECT_EQ("a(1,2)3", tensorstore::StrCat("a", std::complex<float>(1, 2), 3));

  char a = 'a';
  EXPECT_EQ("a", tensorstore::StrCat(a));
}

TEST(StrCat, Enum) {
  EXPECT_EQ("enum", tensorstore::StrCat(OstreamableEnum::value));
  EXPECT_EQ("0", tensorstore::StrCat(PlainEnum::value));
}

TEST(StrCat, Null) { EXPECT_EQ("null", tensorstore::StrCat(nullptr)); }

TEST(StrCat, Tuple) {
  EXPECT_EQ("{1, 2, abc}", tensorstore::StrCat(std::make_tuple(1, 2.0, "abc")));
}

TEST(StrCat, Pair) {
  EXPECT_EQ("{2, abc}", tensorstore::StrCat(std::make_pair(2.0, "abc")));
}

TEST(StrCat, Container) {
  std::vector<int> x{1, 2, 3};
  EXPECT_EQ("{1, 2, 3}", tensorstore::StrCat(x));

  EXPECT_EQ("{1, 2, 3}", tensorstore::StrCat(tensorstore::span(x)));

  std::map<std::string, int> y{{"a", 1}, {"b", 2}};
  EXPECT_EQ("{{a, 1}, {b, 2}}", tensorstore::StrCat(y));
}

TEST(StrCat, Nested) {
  std::vector<std::pair<int, int>> x{{1, 2}, {2, 3}};
  EXPECT_EQ("{{1, 2}, {2, 3}}", tensorstore::StrCat(x));

  std::pair<std::pair<int, int>, std::pair<int, int>> y{{1, 2}, {2, 3}};
  EXPECT_EQ("{{1, 2}, {2, 3}}", tensorstore::StrCat(y));
}

TEST(SpanTest, Ostream) {
  std::ostringstream ostr;
  ostr << tensorstore::span({1, 2, 3});
  EXPECT_EQ("{1, 2, 3}", ostr.str());
}

// TODO: Consider supporting std::optional

}  // namespace
