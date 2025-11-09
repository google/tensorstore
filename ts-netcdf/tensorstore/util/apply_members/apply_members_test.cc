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

#include "tensorstore/util/apply_members/apply_members.h"

#include <array>
#include <complex>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/apply_members/std_array.h"
#include "tensorstore/util/apply_members/std_complex.h"
#include "tensorstore/util/apply_members/std_pair.h"
#include "tensorstore/util/apply_members/std_tuple.h"

namespace {

struct Foo {
  int x, y;
  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.x, x.y);
  };
};

struct Bar {};

struct Baz {
  int x, y;
};

[[maybe_unused]] void TestFooApplyMembers() {
  Foo value;
  tensorstore::ApplyMembers<Foo>::Apply(value, [&](int& x, int& y) {});
}

[[maybe_unused]] void TestBarApplyMembers() {
  Bar value;
  tensorstore::ApplyMembers<Bar>::Apply(value, [&]() {});
}

[[maybe_unused]] void TestTupleApplyMembers() {
  using T = std::tuple<int, double>;
  T value;
  tensorstore::ApplyMembers<T>::Apply(value, [&](int& x, double& y) {});
}

[[maybe_unused]] void TestStdArrayApplyMembers() {
  using T = std::array<int, 3>;
  T value;
  tensorstore::ApplyMembers<T>::Apply(value, [&](int& x, int& y, int& z) {});
}

[[maybe_unused]] void TestArrayApplyMembers() {
  using T = int[3];
  T value;
  tensorstore::ApplyMembers<T>::Apply(value, [&](int& x, int& y, int& z) {});
}

[[maybe_unused]] void TestPairApplyMembers() {
  using T = std::pair<int, double>;
  T value;
  tensorstore::ApplyMembers<T>::Apply(value, [&](int& x, double& y) {});
}

[[maybe_unused]] void TestComplexApplyMembers() {
  using T = std::complex<double>;
  T value;
  tensorstore::ApplyMembers<T>::Apply(value, [&](double& r, double& i) {});
}

static_assert(tensorstore::SupportsApplyMembers<Foo>);
static_assert(tensorstore::SupportsApplyMembers<Bar>);
static_assert(tensorstore::SupportsApplyMembers<std::complex<float>>);
static_assert(tensorstore::SupportsApplyMembers<std::pair<int, double>>);
static_assert(tensorstore::SupportsApplyMembers<std::tuple<>>);
static_assert(tensorstore::SupportsApplyMembers<std::tuple<int>>);
static_assert(tensorstore::SupportsApplyMembers<std::tuple<int, double>>);
static_assert(tensorstore::SupportsApplyMembers<std::array<int, 3>>);
static_assert(!tensorstore::SupportsApplyMembers<Baz>);

}  // namespace
