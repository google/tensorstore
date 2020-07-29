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

#include "tensorstore/internal/json_pprint_python.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using tensorstore::internal_python::PrettyPrintJsonAsPython;
using tensorstore::internal_python::PrettyPrintJsonAsPythonRepr;

TEST(PrettyPrintJsonAsPythonTest, Basic) {
  EXPECT_EQ("None", PrettyPrintJsonAsPython(::nlohmann::json(nullptr)));
  EXPECT_EQ("True", PrettyPrintJsonAsPython(::nlohmann::json(true)));
  EXPECT_EQ("False", PrettyPrintJsonAsPython(::nlohmann::json(false)));
  EXPECT_EQ("'abc'", PrettyPrintJsonAsPython(::nlohmann::json("abc")));
  EXPECT_EQ("1", PrettyPrintJsonAsPython(::nlohmann::json(1)));
  EXPECT_EQ("1.5", PrettyPrintJsonAsPython(::nlohmann::json(1.5)));
  EXPECT_EQ("[1, 2, 3]", PrettyPrintJsonAsPython(::nlohmann::json({1, 2, 3})));
  EXPECT_EQ("[1, 2, 3]",
            PrettyPrintJsonAsPython(::nlohmann::json({1, 2, 3}),
                                    {/*.indent=*/2, /*.width=*/9}));
  EXPECT_EQ(R"([
  1,
  2,
  3,
])",
            PrettyPrintJsonAsPython(::nlohmann::json({1, 2, 3}),
                                    {/*.indent=*/2, /*.width=*/5}));
  EXPECT_EQ("{'a': 1, 'b': 2, 'c': 3}",
            PrettyPrintJsonAsPython(
                ::nlohmann::json({{"a", 1}, {"b", 2}, {"c", 3}})));
  EXPECT_EQ(
      "{'a': 1, 'b': 2, 'c': 3}",
      PrettyPrintJsonAsPython(::nlohmann::json({{"a", 1}, {"b", 2}, {"c", 3}}),
                              {/*.indent=*/2, /*.width=*/24}));
  EXPECT_EQ(
      R"({
  'a': 1,
  'b': 2,
  'c': 3,
})",
      PrettyPrintJsonAsPython(::nlohmann::json({{"a", 1}, {"b", 2}, {"c", 3}}),
                              {/*.indent=*/2, /*.width=*/10}));
  EXPECT_EQ(
      R"({
  'a': 1,
  'b': 2,
  'c': [
    1,
    2,
    3,
    4,
  ],
})",
      PrettyPrintJsonAsPython(
          ::nlohmann::json({{"a", 1}, {"b", 2}, {"c", {1, 2, 3, 4}}}),
          {/*.indent=*/2, /*.width=*/10}));
  EXPECT_EQ(
      R"({
  'a': 1,
  'b': 2,
  'c': [1, 2, 3, 4],
})",
      PrettyPrintJsonAsPython(
          ::nlohmann::json({{"a", 1}, {"b", 2}, {"c", {1, 2, 3, 4}}}),
          {/*.indent=*/2, /*.width=*/21}));
}

TEST(PrettyPrintJsonAsPythonReprTest, Basic) {
  EXPECT_EQ("Foo(None)", PrettyPrintJsonAsPythonRepr(::nlohmann::json(nullptr),
                                                     "Foo(", ")"));
  EXPECT_EQ("Foo(...)",
            PrettyPrintJsonAsPythonRepr(absl::UnknownError(""), "Foo(", ")"));
  EXPECT_EQ(
      R"(Foo({
  'a': 1,
  'b': 2,
  'c': [1, 2, 3, 4],
}))",
      PrettyPrintJsonAsPythonRepr(
          ::nlohmann::json({{"a", 1}, {"b", 2}, {"c", {1, 2, 3, 4}}}), "Foo(",
          ")", {/*.indent=*/2, /*.width=*/21}));
}

}  // namespace
