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

#include "tensorstore/internal/json_binding/rational.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/rational.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Rational;

namespace {

TEST(JsonBindingTest, Simple) {
  tensorstore::TestJsonBinderRoundTrip<Rational<Index>>({
      {{2, 3}, "2/3"},
      {2, 2},
      {1, 1},
      {0, 0},
  });
  tensorstore::TestJsonBinderRoundTripJsonOnly<Rational<Index>>({
      "2/0",
      "3/0",
      "0/0",
  });
  tensorstore::TestJsonBinderRoundTripJsonOnlyInexact<Rational<Index>>({
      {{2, 3}, "2/3"},
  });
  tensorstore::TestJsonBinderFromJson<Rational<Index>>({
      {"abc",
       MatchesStatus(
           absl::StatusCode::kInvalidArgument,
           "Expected number or rational number `a/b`, but received: \"abc\"")},
      {"12a",
       MatchesStatus(
           absl::StatusCode::kInvalidArgument,
           "Expected number or rational number `a/b`, but received: \"12a\"")},
      {"12/a",
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Expected rational number `a/b`, but received: \"12/a\"")},
      {{1},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Array has length 1 but should have length 2")},
      {{1, "a"},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Error parsing value at position 1: "
                     "Expected 64-bit signed integer, but received: \"a\"")},
      {{1, 2, 3},
       MatchesStatus(absl::StatusCode::kInvalidArgument,
                     "Array has length 3 but should have length 2")},
  });
}
}  // namespace
