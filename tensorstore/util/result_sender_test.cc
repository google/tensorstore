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

/// Tests for the `Sender` and `Receiver` interfaces of `Result`.

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/sender_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Result;
using tensorstore::Status;

TEST(ResultReceiverTest, SetCancel) {
  Result<int> result = absl::UnknownError("");
  tensorstore::AnyReceiver<Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_cancel(receiver);
  EXPECT_EQ(result, Result<int>(absl::CancelledError("")));
}

TEST(ResultReceiverTest, SetValue) {
  Result<int> result = absl::UnknownError("");
  tensorstore::AnyReceiver<Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_value(receiver, 3);
  EXPECT_EQ(result, Result<int>(3));
}

TEST(ResultReceiverTest, SetError) {
  Result<int> result = absl::UnknownError("");
  tensorstore::AnyReceiver<Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_error(receiver, absl::UnknownError("message"));
  EXPECT_EQ(result, Result<int>(absl::UnknownError("message")));
}

TEST(ResultSenderTest, SetValue) {
  Result<int> result(3);
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(std::ref(result)),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3"));
}

TEST(ResultSenderTest, SetError) {
  Result<int> result{absl::UnknownError("")};
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(std::ref(result)),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: UNKNOWN: "));
}

TEST(ResultSenderTest, SetCancel) {
  Result<int> result{absl::CancelledError("")};
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(std::ref(result)),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

}  // namespace
