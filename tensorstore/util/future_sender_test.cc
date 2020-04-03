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

/// Tests for the `Sender` and `Receiver` interfaces of `Future` and `Promise`.

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/sender_testutil.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Promise;
using tensorstore::PromiseFuturePair;
using tensorstore::Result;
using tensorstore::Status;

TEST(PromiseReceiverTest, SetCancel) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_cancel(pair.promise);
  EXPECT_EQ(pair.future.result(), Result<int>(absl::CancelledError("")));
}

TEST(PromiseReceiverTest, AnyReceiverSetCancel) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_cancel(
      tensorstore::AnyReceiver<Status, int>(pair.promise));
  EXPECT_EQ(pair.future.result(), Result<int>(absl::CancelledError("")));
}

TEST(PromiseReceiverTest, SetValue) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_value(pair.promise, 3);
  EXPECT_EQ(pair.future.result(), Result<int>(3));
}

// Tests that calling set_cancel after set_value has no effect.
TEST(PromiseReceiverTest, SetValueThenSetCancel) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_value(pair.promise, 3);
  tensorstore::execution::set_cancel(pair.promise);
  EXPECT_EQ(pair.future.result(), Result<int>(3));
}

TEST(PromiseReceiverTest, AnyReceiverSetValue) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_value(
      tensorstore::AnyReceiver<Status, int>(pair.promise), 3);
  EXPECT_EQ(pair.future.result(), Result<int>(3));
}

TEST(PromiseReceiverTest, SetError) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_error(
      tensorstore::AnyReceiver<Status, int>(pair.promise),
      absl::UnknownError("message"));
  EXPECT_EQ(pair.future.result(), Result<int>(absl::UnknownError("message")));
}

TEST(PromiseReceiverTest, AnyReceiverSetError) {
  auto pair = PromiseFuturePair<int>::Make();
  tensorstore::execution::set_error(pair.promise,
                                    absl::UnknownError("message"));
  EXPECT_EQ(pair.future.result(), Result<int>(absl::UnknownError("message")));
}

TEST(FutureSenderTest, SetValue) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log1, log2;
  tensorstore::execution::submit(pair.future,
                                 tensorstore::LoggingReceiver{&log1});
  tensorstore::execution::submit(pair.future,
                                 tensorstore::LoggingReceiver{&log2});
  EXPECT_THAT(log1, ::testing::ElementsAre());
  EXPECT_THAT(log2, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(3);
  EXPECT_THAT(log1, ::testing::ElementsAre("set_value: 3"));
  EXPECT_THAT(log2, ::testing::ElementsAre("set_value: 3"));
}

TEST(FutureSenderTest, AnySenderSetValue) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(pair.future),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(3);
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3"));
}

TEST(FutureSenderTest, SetError) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log;
  tensorstore::execution::submit(pair.future,
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(absl::UnknownError(""));
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: UNKNOWN: "));
}

TEST(FutureSenderTest, AnySenderSetError) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(pair.future),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(absl::UnknownError(""));
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: UNKNOWN: "));
}

TEST(FutureSenderTest, SetCancel) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log;
  tensorstore::execution::submit(pair.future,
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(absl::CancelledError(""));
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(FutureSenderTest, AnySenderSetCancel) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  pair.promise.ExecuteWhenForced([&](Promise<int>) { forced = true; });
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<Status, int>(pair.future),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_TRUE(forced);
  pair.promise.SetResult(absl::CancelledError(""));
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(MakeSenderFutureTest, SetValue) {
  auto future =
      tensorstore::MakeSenderFuture<int>(tensorstore::ValueSender<int>{3});
  EXPECT_FALSE(future.ready());
  EXPECT_EQ(future.result(), Result<int>(3));
}

TEST(MakeSenderFutureTest, SetError) {
  auto future = tensorstore::MakeSenderFuture<int>(
      tensorstore::ErrorSender<Status>{absl::UnknownError("")});
  EXPECT_FALSE(future.ready());
  EXPECT_EQ(future.result(), Result<int>(absl::UnknownError("")));
}

TEST(MakeSenderFutureTest, SetCancel) {
  auto future = tensorstore::MakeSenderFuture<int>(tensorstore::CancelSender{});
  EXPECT_FALSE(future.ready());
  EXPECT_EQ(future.result(), Result<int>(absl::CancelledError("")));
}

}  // namespace
