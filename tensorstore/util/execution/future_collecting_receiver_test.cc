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

#include "tensorstore/util/execution/future_collecting_receiver.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::CollectFlowSenderIntoFuture;
using ::tensorstore::MatchesStatus;

TEST(CollectingSenderTest, Success) {
  std::vector<int> input{1, 2, 3, 4};

  EXPECT_THAT(CollectFlowSenderIntoFuture<std::vector<int>>(
                  tensorstore::RangeFlowSender<tensorstore::span<int>>{input})
                  .result(),
              ::testing::Optional(::testing::ElementsAreArray(input)));
}

TEST(CollectingSenderTest, Error) {
  EXPECT_THAT(
      CollectFlowSenderIntoFuture<std::vector<int>>(
          tensorstore::FlowSingleSender<tensorstore::ErrorSender<absl::Status>>{
              absl::UnknownError("abc")})
          .result(),
      MatchesStatus(absl::StatusCode::kUnknown, "abc"));
}

}  // namespace
