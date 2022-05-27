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

#include "tensorstore/internal/json_binding/std_tuple.h"

#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/status_testutil.h"

namespace jb = tensorstore::internal_json_binding;

namespace {


TEST(TupleDefaultJsonBinderTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTrip<std::pair<int, int>>({
      {{5, 5}, {5, 5}},
      {{5, 3}, {5, 3}},
  });
  tensorstore::TestJsonBinderRoundTrip<std::tuple<int, int, std::string>>({
      {{5, 5, "a"}, {5, 5, "a"}},
      {{5, 3, "b"}, {5, 3, "b"}},
  });
}

TEST(TupleJsonBinderTest, RoundTrip) {
  const auto binder =
      jb::Tuple(jb::Integer<int>(0, 9), jb::Integer<int>(10, 19));
  tensorstore::TestJsonBinderRoundTrip<std::pair<int, int>>(
      {
          {{5, 15}, {5, 15}},
          {{5, 13}, {5, 13}},
      },
      binder);
}

TEST(HeterogeneousArrayJsonBinderTest, RoundTrip) {
  struct X {
    int a;
    std::string b;
  };
  tensorstore::TestJsonBinderRoundTripJsonOnly<X>(
      {
          {5, "a"},
          {5, "b"},
      },
      jb::HeterogeneousArray(jb::Projection<&X::a>(), jb::Projection<&X::b>()));
}

}  // namespace
