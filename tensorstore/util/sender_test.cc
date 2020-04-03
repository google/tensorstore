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

#include "tensorstore/util/sender.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/sender_testutil.h"

namespace {

TEST(NullReceiverTest, SetDone) {
  tensorstore::NullReceiver receiver;
  tensorstore::execution::set_done(receiver);
}

TEST(NullReceiverTest, SetValue) {
  tensorstore::NullReceiver receiver;
  tensorstore::execution::set_value(receiver, 3, 4);
}

TEST(NullReceiverTest, SetError) {
  tensorstore::NullReceiver receiver;
  tensorstore::execution::set_value(receiver, 10);
}

TEST(AnyReceiverTest, NullSetCancel) {
  tensorstore::AnyReceiver<int> receiver;
  tensorstore::execution::set_cancel(receiver);
}

TEST(AnyReceiverTest, NullSetValue) {
  tensorstore::AnyReceiver<int, std::string> receiver;
  tensorstore::execution::set_value(receiver, "message");
}

TEST(AnyReceiverTest, NullSetError) {
  tensorstore::AnyReceiver<int, std::string> receiver;
  tensorstore::execution::set_error(receiver, 3);
}

TEST(CancelSenderTest, Basic) {
  std::vector<std::string> log;
  tensorstore::execution::submit(tensorstore::CancelSender{},
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(CancelSenderTest, AnySender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(tensorstore::CancelSender{}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(ErrorSenderTest, Basic) {
  std::vector<std::string> log;
  tensorstore::execution::submit(tensorstore::ErrorSender<int>{3},
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

TEST(ErrorSenderTest, AnySender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(tensorstore::ErrorSender<int>{3}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

TEST(ValueSenderTest, Basic) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::ValueSender<int, std::string>{3, "hello"},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

TEST(ValueSenderTest, AnySender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int, int, std::string>(
          tensorstore::ValueSender<int, std::string>{3, "hello"}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

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

struct QueueExecutor {
  std::vector<tensorstore::ExecutorTask>* queue;
  void operator()(tensorstore::ExecutorTask task) const {
    queue->push_back(std::move(task));
  }
};

TEST(SenderWithExecutorTest, SetValue) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::SenderWithExecutor<
          tensorstore::ValueSender<int, std::string>>{executor, {3, "hello"}},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

TEST(SenderWithExecutorTest, AnySenderSetValue) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int, int, std::string>(
          tensorstore::SenderWithExecutor<
              tensorstore::ValueSender<int, std::string>>{executor,
                                                          {3, "hello"}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

TEST(SenderWithExecutorTest, SetError) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::SenderWithExecutor<tensorstore::ErrorSender<int>>{executor,
                                                                     {3}},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

TEST(SenderWithExecutorTest, AnySenderSetError) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(
          tensorstore::SenderWithExecutor<tensorstore::ErrorSender<int>>{
              executor, {3}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

TEST(SenderWithExecutorTest, SetCancel) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::SenderWithExecutor<tensorstore::CancelSender>{executor},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(SenderWithExecutorTest, AnySenderSetCancel) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(
          tensorstore::SenderWithExecutor<tensorstore::CancelSender>{executor}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
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
