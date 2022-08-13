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

#include <any>
#include <utility>

#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::MakeResult;
using ::tensorstore::Result;

struct W {
  explicit W(Result<int>) {}
};

struct X {
  X(Result<int>) {}
};

struct Z {
  Z& operator=(Result<int>) { return *this; }
};

void BasicNonCompile() {
  Result<void> v = MakeResult();
  EXPECT_NON_COMPILE("value_or", v.value_or());
  EXPECT_NON_COMPILE("pointer", *v);

  Result<int> int_result(3);

  // Result<W> requires use of the in_place constructor.
  EXPECT_NON_COMPILE("no matching", Result<W>{int_result});
  EXPECT_NON_COMPILE("no viable conversion", Result<W> y = int_result);

  // Result<X> requires use of the in_place constructor.
  EXPECT_NON_COMPILE("no matching", Result<X>{int_result});
  EXPECT_NON_COMPILE("no viable conversion", Result<X> y = int_result);

  // Result<Z> is not assignable.
  EXPECT_NON_COMPILE("no viable overload", Result<Z> z; z = int_result);

  // Ambiguous cases
  EXPECT_NON_COMPILE("no matching", Result<std::any> s(Result<int>(3)););
  EXPECT_NON_COMPILE("no matching", Result<std::any> s(int_result););
  EXPECT_NON_COMPILE("no viable overload", Result<std::any> s; s = int_result;);

  /// FIXME: The semantics of Result<const X> need additional thought.
  EXPECT_NON_COMPILE("static_assert", Result<const int>{std::in_place});

  /// FIXME: Properly specialize on <const void> vs. <void>.
  /// This is in addition to enabling const, above.
  EXPECT_NON_COMPILE("static_assert", Result<const void>{std::in_place});

  // Result<Result<>> Omits some constructor forms.
  EXPECT_NON_COMPILE("no matching", Result<Result<int>> r(Result<int>(3)););
  EXPECT_NON_COMPILE("no viable conversion",
                     Result<Result<int>> r = Result<int>(3););

  // Result<std::unique_ptr<int>> copy-ctor/copy-assignment unavailable.
  std::unique_ptr<int> ptr;
  EXPECT_NON_COMPILE("no matching", Result<std::unique_ptr<int>> r(ptr););
  EXPECT_NON_COMPILE("no viable", Result<std::unique_ptr<int>> r; r = ptr;);
}

}  // namespace

int main() { BasicNonCompile(); }
