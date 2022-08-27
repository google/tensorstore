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

#include "tensorstore/util/execution/sender.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/executor.h"

namespace {

// Verify execution.h detected_t mechanism.
template <typename T, typename... Arg>
using trait_has_submit =
    decltype(std::declval<T&&>().submit(std::declval<Arg>()...));
template <typename... Arg>
using trait_has_adl_submit = decltype(submit(std::declval<Arg>()...));

static_assert(!tensorstore::internal_execution::detected_t<
              trait_has_submit, tensorstore::NullSender&, int>::value);
static_assert(tensorstore::internal_execution::detected_t<
              trait_has_adl_submit, tensorstore::NullSender&, int>::value);

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
  tensorstore::execution::set_error(receiver, 10);
}

TEST(CancelSenderTest, Basic) {
  std::vector<std::string> log;
  tensorstore::execution::submit(tensorstore::CancelSender{},
                                 tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

TEST(ErrorSenderTest, Basic) {
  std::vector<std::string> log;
  tensorstore::execution::submit(tensorstore::ErrorSender<int>{3},
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

TEST(SenderWithExecutorTest, SetValue) {
  std::vector<tensorstore::ExecutorTask> queue;
  std::vector<std::string> log;
  QueueExecutor executor{&queue};
  tensorstore::execution::submit(
      SenderWithExecutor<tensorstore::ValueSender<int, std::string>,
                         tensorstore::Executor>{executor, {3, "hello"}},
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
      SenderWithExecutor<tensorstore::ErrorSender<int>, tensorstore::Executor>{
          executor, {3}},
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
      SenderWithExecutor<tensorstore::CancelSender, tensorstore::Executor>{
          executor},
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre());
  EXPECT_EQ(1, queue.size());
  queue[0]();
  EXPECT_THAT(log, ::testing::ElementsAre("set_cancel"));
}

}  // namespace
