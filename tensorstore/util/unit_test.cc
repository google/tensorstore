// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/util/unit.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/unit.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::TestJsonBinderRoundTrip;
using ::tensorstore::TestJsonBinderRoundTripJsonOnlyInexact;
using ::tensorstore::Unit;
using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(UnitTest, DefaultConstruct) {
  Unit u;
  EXPECT_EQ(1, u.multiplier);
  EXPECT_EQ("", u.base_unit);
}

TEST(UnitTest, Compare) {
  Unit a(5, "nm");
  Unit b(5.5, "nm");
  Unit c(5, "um");
  Unit d;
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(b, c);
  EXPECT_NE(b, d);
  EXPECT_NE(c, d);
}

TEST(UnitTest, Ostream) {
  EXPECT_EQ("5.5 nm", tensorstore::StrCat(Unit(5.5, "nm")));
  EXPECT_EQ("nm", tensorstore::StrCat(Unit(1, "nm")));
  EXPECT_EQ("5", tensorstore::StrCat(Unit(5, "")));
  EXPECT_EQ("1", tensorstore::StrCat(Unit(1, "")));
}

TEST(UnitTest, MultiplierBaseUnit) {
  Unit u = {5, "nm"};
  EXPECT_EQ(5, u.multiplier);
  EXPECT_EQ("nm", u.base_unit);
}

TEST(UnitTest, Unit) {
  EXPECT_EQ(Unit(4, "nm"), Unit("4nm"));
  EXPECT_EQ(Unit(4, "nm"), Unit("4.nm"));
  EXPECT_EQ(Unit(4e-3, "nm"), Unit("4e-3nm"));
  EXPECT_EQ(Unit(.4, "nm"), Unit(".4nm"));
  EXPECT_EQ(Unit(.4, "nm"), Unit(".4 nm"));
  EXPECT_EQ(Unit(.4, "nm"), Unit(" .4 nm"));
  EXPECT_EQ(Unit(.4, "nm"), Unit(" .4 nm "));
  EXPECT_EQ(Unit(4e-3, "nm"), Unit("+4e-3nm"));
  EXPECT_EQ(Unit(-4e-3, "nm"), Unit("-4e-3nm"));
  EXPECT_EQ(Unit(4.5, "nm"), Unit("4.5nm"));
  EXPECT_EQ(Unit(1, "nm"), Unit("nm"));
  EXPECT_EQ(Unit(4, ""), Unit("4"));
  EXPECT_EQ(Unit(1, ""), Unit(""));
  // "offset" units supported by udunits2 library.  Here we just confirm that
  // the number is not confused for a leading multiplier.
  EXPECT_EQ(Unit(3, "nm @ 50"), Unit("3 nm @ 50"));
}

TEST(UnitTest, JsonRoundTrip) {
  TestJsonBinderRoundTrip<Unit>({
      {Unit(4, "nm"), {4, "nm"}},
      {Unit(4.5, "nm"), {4.5, "nm"}},
      {Unit(4.5, ""), {4.5, ""}},
  });
}

TEST(UnitTest, JsonRoundTripInexact) {
  TestJsonBinderRoundTripJsonOnlyInexact<Unit>({
      {"4nm", {4, "nm"}},
      {4, {4, ""}},
      {"nm", {1, "nm"}},
  });
}

TEST(SerializationTest, Basic) {
  TestSerializationRoundTrip(Unit("4nm"));
  TestSerializationRoundTrip(Unit("4"));
  TestSerializationRoundTrip(Unit("nm"));
  TestSerializationRoundTrip(Unit(""));
}

}  // namespace
