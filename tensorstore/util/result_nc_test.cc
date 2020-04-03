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

/// Non-compile test for result.h.

#include <utility>

#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::MakeResult;
using tensorstore::Result;

struct W {
  explicit W(Result<int>) {}
};

struct X {
  X(Result<int>) {}
};

void BasicNonCompile() {
  Result<void> v = MakeResult();
  EXPECT_NON_COMPILE("value_or", v.value_or());
  EXPECT_NON_COMPILE("pointer", *v);

  Result<int> int_result(3);

  // Result<W> requires use of the in_place constructor.
  EXPECT_NON_COMPILE("", Result<W>{int_result});
  EXPECT_NON_COMPILE("no viable conversion", Result<W> y = int_result);

  // Result<X> requires use of the in_place constructor.
  EXPECT_NON_COMPILE("", Result<X>{int_result});
  EXPECT_NON_COMPILE("no viable conversion", Result<X> y = int_result);

  // Result<void> lacks comparison operators.
  Result<void> r{absl::in_place};
  EXPECT_NON_COMPILE("", r == Result<void>(absl::in_place));

  /// FIXME: The semantics of Result<const X> need additional thought.
  EXPECT_NON_COMPILE("", Result<const int>{absl::in_place});

  /// FIXME: Properly specialize on <const void> vs. <void>.
  /// This is in addition to enabling const, above.
  EXPECT_NON_COMPILE("", Result<const void>{absl::in_place});
}

}  // namespace

int main() { BasicNonCompile(); }
