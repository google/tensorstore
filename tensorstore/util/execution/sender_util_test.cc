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

#include "tensorstore/util/execution/sender_util.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_testutil.h"

namespace {

TEST(FlowSingleSenderTest, SetValue) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::FlowSingleSender<tensorstore::ValueSender<int, std::string>>{
          {3, "hello"}},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_value: 3, hello",
                                          "set_done", "set_stopping"));
}

TEST(FlowSingleSenderTest, AnyFlowSenderSetValue) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int, int, std::string>(
          tensorstore::FlowSingleSender<
              tensorstore::ValueSender<int, std::string>>{{3, "hello"}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_value: 3, hello",
                                          "set_done", "set_stopping"));
}

TEST(FlowSingleSenderTest, SetError) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::FlowSingleSender<tensorstore::ErrorSender<int>>{{3}},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_error: 3",
                                          "set_stopping"));
}

TEST(FlowSingleSenderTest, AnyFlowSenderSetError) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int>(
          tensorstore::FlowSingleSender<tensorstore::ErrorSender<int>>{{3}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_error: 3",
                                          "set_stopping"));
}

TEST(FlowSingleSenderTest, SetCancel) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::FlowSingleSender<tensorstore::CancelSender>{},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_done", "set_stopping"));
}

TEST(FlowSingleSenderTest, AnyFlowSenderSetCancel) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int>(
          tensorstore::FlowSingleSender<tensorstore::CancelSender>{}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_done", "set_stopping"));
}

TEST(RangeFlowSenderTest, Basic) {
  std::vector<int> values{1, 2, 3};
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int, int>(
          tensorstore::RangeFlowSender<std::vector<int>&>{values}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_value: 1",
                                          "set_value: 2", "set_value: 3",
                                          "set_done", "set_stopping"));
}

TEST(RangeFlowSenderTest, CancelImmediately) {
  std::vector<int> values{1, 2, 3};
  std::vector<std::string> log;
  struct Receiver : public tensorstore::LoggingReceiver {
    tensorstore::AnyCancelReceiver cancel;
    void set_starting(tensorstore::AnyCancelReceiver cancel) {
      this->tensorstore::LoggingReceiver::set_starting({});
      cancel();
    }
  };
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int, int>(
          tensorstore::RangeFlowSender<std::vector<int>&>{values}),
      Receiver{{&log}});
  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_done", "set_stopping"));
}

TEST(RangeFlowSenderTest, Cancel) {
  std::vector<int> values{1, 2, 3};
  std::vector<std::string> log;
  struct Receiver : public tensorstore::LoggingReceiver {
    tensorstore::AnyCancelReceiver cancel;
    void set_starting(tensorstore::AnyCancelReceiver cancel) {
      this->cancel = std::move(cancel);
      this->tensorstore::LoggingReceiver::set_starting({});
    }

    void set_value(int value) {
      this->tensorstore::LoggingReceiver::set_value(value);
      if (value == 2) {
        this->cancel();
      }
    }
  };
  tensorstore::execution::submit(
      tensorstore::AnyFlowSender<int, int>(
          tensorstore::RangeFlowSender<std::vector<int>&>{values}),
      Receiver{{&log}});
  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_value: 1",
                                  "set_value: 2", "set_done", "set_stopping"));
}

}  // namespace
