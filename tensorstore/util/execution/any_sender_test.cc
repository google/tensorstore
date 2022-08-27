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

#include "tensorstore/util/execution/any_sender.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/executor.h"

namespace {

TEST(AnySenderTest, Construct) {
  tensorstore::AnySender<int, std::string> sender(tensorstore::CancelSender{});
}

TEST(AnySenderTest, Assignment) {
  tensorstore::AnySender<int, std::string> sender;
  sender = tensorstore::CancelSender{};
}

TEST(AnySenderTest, Submit) {
  tensorstore::AnySender<int, std::string> sender;
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(tensorstore::NullSender{}),
      tensorstore::NullReceiver{});
}

TEST(AnySenderTest, CancelSender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(tensorstore::CancelSender{}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(AnySenderTest, ErrorSender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(tensorstore::ErrorSender<int>{3}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

TEST(AnySenderTest, ValueSender) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::AnySender<int, int, std::string>(
          tensorstore::ValueSender<int, std::string>{3, "hello"}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

/// Sender that adapts an existing `sender` to invoke its `submit` function with
/// the specified `executor`.
template <typename Sender, typename Executor>
struct SenderWithExecutor {
  Executor executor;
  Sender sender;
  template <typename Receiver>
  void submit(Receiver receiver) {
    struct Callback {
      Sender sender;
      Receiver receiver;
      void operator()() {
        tensorstore::execution::submit(sender, std::move(receiver));
      }
    };
    executor(Callback{std::move(sender), std::move(receiver)});
  }
};

struct QueueExecutor {
  std::vector<tensorstore::ExecutorTask>* queue;
  void operator()(tensorstore::ExecutorTask task) const {
    queue->push_back(std::move(task));
  }
};

TEST(AnySenderWithExecutor, SetValue) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int, int, std::string>(
          SenderWithExecutor<tensorstore::ValueSender<int, std::string>,
                             tensorstore::Executor>{executor, {3, "hello"}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 3, hello"));
}

TEST(AnySenderWithExecutor, SetCancel) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(
          SenderWithExecutor<tensorstore::CancelSender, tensorstore::Executor>{
              executor}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(AnySenderWithExecutor, SetError) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      tensorstore::AnySender<int>(
          SenderWithExecutor<tensorstore::ErrorSender<int>,
                             tensorstore::Executor>{executor, {3}}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 3"));
}

// -------------------------------------------------------------------------

TEST(AnyFlowSenderTest, Construct) {
  tensorstore::AnyFlowSender<int, std::string> sender(
      tensorstore::NullSender{});
}

TEST(AnyFlowSenderTest, Assignment) {
  tensorstore::AnyFlowSender<int, std::string> sender;
  sender = tensorstore::NullSender{};
}

TEST(AnyFlowSenderTest, Submit) {
  tensorstore::AnyFlowSender<int, std::string> sender;
  tensorstore::execution::submit(std::move(sender),
                                 tensorstore::NullReceiver{});
}

TEST(AnyFlowSenderTest, ValueSender) {
  std::vector<std::string> log;
  tensorstore::AnyFlowSender<int, std::string> sender(
      tensorstore::ValueSender("A"));
  tensorstore::execution::submit(std::move(sender),
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: A"));
}

TEST(AnyFlowSenderTest, ErrorSender) {
  std::vector<std::string> log;
  tensorstore::AnyFlowSender<int, std::string> sender(
      tensorstore::ErrorSender<int>{4});
  tensorstore::execution::submit(std::move(sender),
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 4"));
}

struct MySender {
  template <typename Receiver>
  void submit(Receiver receiver) {
    tensorstore::execution::set_starting(receiver, []() {});
    tensorstore::execution::set_value(receiver, "B");
    tensorstore::execution::set_value(receiver, "C");
    tensorstore::execution::set_done(receiver);
    tensorstore::execution::set_stopping(receiver);
  }
};

TEST(AnyFlowSenderTest, MySender) {
  std::vector<std::string> log;

  tensorstore::AnyFlowSender<int, std::string> sender(MySender{});
  tensorstore::execution::submit(std::move(sender),
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(
      log, ::testing::ElementsAre("set_starting", "set_value: B",
                                  "set_value: C", "set_done", "set_stopping"));
}

}  // namespace
