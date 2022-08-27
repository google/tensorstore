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

#include "tensorstore/util/execution/any_receiver.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_testutil.h"

namespace {

TEST(AnyReceiverTest, Construct) {
  tensorstore::AnyReceiver<int, std::string> receiver(
      tensorstore::NullReceiver{});
}

TEST(AnyReceiverTest, Assignment) {
  tensorstore::AnyReceiver<int, std::string> receiver;
  receiver = tensorstore::NullReceiver{};

  {
    tensorstore::NullReceiver tmp{};
    receiver = tmp;
  }
}

TEST(AnyReceiverTest, NullSetValue) {
  tensorstore::AnyReceiver<int, std::string> receiver;
  tensorstore::execution::set_value(receiver, "message");
}

TEST(AnyReceiverTest, NullSetError) {
  tensorstore::AnyReceiver<int, std::string> receiver;
  tensorstore::execution::set_error(receiver, 3);
}

TEST(AnyReceiverTest, NullSetCancel) {
  tensorstore::AnyReceiver<int> receiver;
  tensorstore::execution::set_cancel(receiver);
}

TEST(AnyReceiverTest, LoggingSetValue) {
  std::vector<std::string> log;
  tensorstore::AnyReceiver<int, std::string> receiver(
      tensorstore::LoggingReceiver{&log});
  tensorstore::execution::set_value(receiver, "ok");

  EXPECT_THAT(log, ::testing::ElementsAre("set_value: ok"));
}

TEST(AnyReceiverTest, SetErrorInt) {
  std::vector<std::string> log;
  tensorstore::AnyReceiver<int, std::string> receiver(
      tensorstore::LoggingReceiver{&log});
  tensorstore::execution::set_error(receiver, 5);

  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 5"));
}

TEST(AnyReceiverTest, SetCancel) {
  std::vector<std::string> log;
  tensorstore::AnyReceiver<int, std::string> receiver(
      tensorstore::LoggingReceiver{&log});
  tensorstore::execution::set_cancel(receiver);

  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

// -------------------------------------------------------------------------

TEST(AnyFlowReceiver, Construct) {
  tensorstore::AnyFlowReceiver<int, std::string> receiver(
      tensorstore::NullReceiver{});
}

TEST(AnyFlowReceiver, Assignment) {
  tensorstore::AnyFlowReceiver<int, std::string> receiver;
  receiver = tensorstore::NullReceiver{};

  {
    tensorstore::NullReceiver tmp{};
    receiver = tmp;
  }
}

TEST(AnyFlowReceiver, NullSetStarting) {
  tensorstore::AnyFlowReceiver<int> receiver;
  tensorstore::execution::set_starting(receiver, []() {});
}

TEST(AnyFlowReceiver, NullSetValue) {
  tensorstore::AnyFlowReceiver<int, std::string> receiver;
  tensorstore::execution::set_value(receiver, "messaage");
}

TEST(AnyFlowReceiver, NullSetError) {
  tensorstore::AnyFlowReceiver<int, std::string> receiver;
  tensorstore::execution::set_error(receiver, 3);
}

TEST(AnyFlowReceiver, NullSetDone) {
  tensorstore::AnyFlowReceiver<int> receiver;
  tensorstore::execution::set_done(receiver);
}

TEST(AnyFlowReceiver, NullSetStopping) {
  tensorstore::AnyFlowReceiver<int> receiver;
  tensorstore::execution::set_stopping(receiver);
}

TEST(AnyFlowReceiver, LoggingSetValue) {
  std::vector<std::string> log;
  tensorstore::AnyFlowReceiver<int, std::string> receiver(
      tensorstore::LoggingReceiver{&log});

  tensorstore::execution::set_starting(receiver, []() {});
  tensorstore::execution::set_value(receiver, "A");
  tensorstore::execution::set_value(receiver, "B");
  tensorstore::execution::set_done(receiver);
  tensorstore::execution::set_stopping(receiver);

  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_value: A",
                                  "set_value: B", "set_done", "set_stopping"));
}

TEST(AnyFlowReceiver, LoggingSetError) {
  std::vector<std::string> log;
  tensorstore::AnyFlowReceiver<int, std::string> receiver(
      tensorstore::LoggingReceiver{&log});

  tensorstore::execution::set_starting(receiver, []() {});
  tensorstore::execution::set_value(receiver, "A");
  tensorstore::execution::set_error(receiver, 5);
  tensorstore::execution::set_done(receiver);
  tensorstore::execution::set_stopping(receiver);

  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_value: A",
                                  "set_error: 5", "set_done", "set_stopping"));
}

}  // namespace
