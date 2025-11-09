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

#include "tensorstore/util/execution/result_sender.h"  // IWYU pragma: keep

#include <functional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Result;
using ::tensorstore::StatusIs;

TEST(ResultReceiverTest, SetCancel) {
  Result<int> result = absl::InternalError("");
  tensorstore::AnyReceiver<absl::Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_cancel(receiver);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kCancelled));
}

TEST(ResultReceiverTest, SetValue) {
  Result<int> result = absl::InternalError("");
  tensorstore::AnyReceiver<absl::Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_value(receiver, 3);
  EXPECT_EQ(result, Result<int>(3));
}

TEST(ResultReceiverTest, SetError) {
  Result<int> result = absl::InternalError("");
  tensorstore::AnyReceiver<absl::Status, int> receiver{std::ref(result)};
  tensorstore::execution::set_error(receiver, absl::UnknownError("message"));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnknown, "message"));
}

TEST(ResultSenderTest, SetValue) {
  Result<int> result(3);
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<absl::Status, int>(result),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3"));
}

TEST(ResultSenderTest, SetError) {
  Result<int> result{absl::UnknownError("")};
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<absl::Status, int>(result),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: UNKNOWN: "));
}

TEST(ResultSenderTest, SetCancel) {
  Result<int> result{absl::CancelledError("")};
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<absl::Status, int>(result),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

}  // namespace
