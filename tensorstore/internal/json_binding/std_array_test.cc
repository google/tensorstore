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

#include "tensorstore/internal/json_binding/std_array.h"

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::MatchesStatus;

namespace jb = tensorstore::internal_json_binding;

namespace {

TEST(JsonBindingTest, Array) {
  const auto binder = jb::Array();
  tensorstore::TestJsonBinderRoundTrip<std::vector<int>>(
      {
          {{1, 2, 3}, {1, 2, 3}},
      },
      binder);
  tensorstore::TestJsonBinderFromJson<std::vector<int>>(
      {
          {{1, 2, "a"},
           MatchesStatus(
               absl::StatusCode::kInvalidArgument,
               "Error parsing value at position 2: Expected integer .*")},
      },
      binder);
}

TEST(JsonBindingTest, FixedSizeArray) {
  const auto binder = jb::FixedSizeArray();
  tensorstore::TestJsonBinderRoundTrip<std::array<int, 3>>(
      {
          {{{1, 2, 3}}, {1, 2, 3}},
      },
      binder);
  tensorstore::TestJsonBinderFromJson<std::array<int, 3>>(
      {

          {{1, 2, 3, 4},
           MatchesStatus(absl::StatusCode::kInvalidArgument,
                         "Array has length 4 but should have length 3")},
      },
      binder);
}

}  // namespace
