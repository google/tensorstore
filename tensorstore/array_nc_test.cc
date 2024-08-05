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

/// Non-compile test for array.h.

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::Index;

void FullIndexing() {
  tensorstore::SharedArray<int, 2> x =
      tensorstore::MakeArray<int>({{1, 2}, {3, 4}});
  static_cast<void>(x(0, 1));
  static_cast<void>(x({0, 1}));
  static_cast<void>(x(tensorstore::span<const Index, 2>({0, 1})));

  EXPECT_NON_COMPILE("double", x({1.1, 2.2}));
  EXPECT_NON_COMPILE("no matching function", x());
  EXPECT_NON_COMPILE("IsCompatibleFullIndexPack", x(1, 2, 3));
  EXPECT_NON_COMPILE("IsCompatibleFullIndexPack", x(1.1, 2, 3));
  EXPECT_NON_COMPILE("IsCompatibleFullIndexPack", x());
  EXPECT_NON_COMPILE("template argument", x({}));
  EXPECT_NON_COMPILE("RankConstraint::EqualOrUnspecified", x({1, 2, 3}));
  EXPECT_NON_COMPILE("IsCompatibleFullIndexVector",
                     x(tensorstore::span<const Index, 3>({1, 2, 3})));
}

void PartialIndexing() {
  tensorstore::SharedArray<int, 2> x =
      tensorstore::MakeArray<int>({{1, 2}, {3, 4}});
  static_cast<void>(x[0]);
  static_cast<void>(x[0][1]);
  static_cast<void>(x[{0, 1}]);
  static_cast<void>(x[tensorstore::span<const Index, 2>({0, 1})]);

  EXPECT_NON_COMPILE("GreaterOrUnspecified", x[0][0][0]);
  EXPECT_NON_COMPILE("no viable overloaded operator\\[\\]", x[{0, 0, 0}]);
  EXPECT_NON_COMPILE("no viable overloaded operator\\[\\]",
                     x[tensorstore::span<const Index, 3>({0, 0, 0})]);
}

}  // namespace

int main() {
  FullIndexing();
  PartialIndexing();
}
