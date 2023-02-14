// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/json_binding/raw_bytes_hex.h"

#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/status_testutil.h"

namespace jb = tensorstore::internal_json_binding;

namespace {

using ::tensorstore::MatchesStatus;

TEST(RawBytesHexTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTrip<std::array<unsigned char, 3>>(
      {
          {{{1, 2, 0xab}}, "0102ab"},
      },
      jb::RawBytesHex);
  tensorstore::TestJsonBinderRoundTripJsonOnlyInexact<
      std::array<unsigned char, 3>>(
      {
          {"0102AB", "0102ab"},
      },
      jb::RawBytesHex);
}

TEST(RawBytesHexTest, Invalid) {
  tensorstore::TestJsonBinderFromJson<std::array<unsigned char, 3>>(
      {
          {1,
           MatchesStatus(absl::StatusCode::kInvalidArgument,
                         "Expected string with 6 hex digits, but received: 1")},
          {"0102zb", MatchesStatus(absl::StatusCode::kInvalidArgument,
                                   "Expected string with 6 hex "
                                   "digits, but received: \"0102zb\"")},
      },
      jb::RawBytesHex);
}

}  // namespace
