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

#include "tensorstore/util/execution/sync_flow_sender.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/thread.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"

namespace {

struct ConcurrentSender {
  std::size_t num_threads;
  bool error;
  template <typename Receiver>
  void submit(Receiver receiver) {
    tensorstore::execution::set_starting(receiver, [] {});
    std::vector<tensorstore::internal::Thread> threads;
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back(tensorstore::internal::Thread(
          {"sender"},
          [i, &receiver] { tensorstore::execution::set_value(receiver, i); }));
    }
    for (auto& thread : threads) thread.Join();
    if (error) {
      tensorstore::execution::set_error(receiver, 3);
    } else {
      tensorstore::execution::set_done(receiver);
    }
    tensorstore::execution::set_stopping(receiver);
  }
};

TEST(SyncFlowSender, Values) {
  std::vector<std::string> log;
  const std::size_t num_threads = 10;
  tensorstore::execution::submit(
      tensorstore::MakeSyncFlowSender(
          ConcurrentSender{num_threads, /*.error=*/false}),
      tensorstore::LoggingReceiver{&log});
  ASSERT_EQ(num_threads + 3, log.size());
  EXPECT_EQ("set_starting", log[0]);
  EXPECT_EQ("set_done", log[log.size() - 2]);
  EXPECT_EQ("set_stopping", log[log.size() - 1]);
  EXPECT_THAT(
      log, ::testing::UnorderedElementsAre(
               "set_starting", "set_value: 0", "set_value: 1", "set_value: 2",
               "set_value: 3", "set_value: 4", "set_value: 5", "set_value: 6",
               "set_value: 7", "set_value: 8", "set_value: 9", "set_done",
               "set_stopping"));
}

TEST(SyncFlowSender, Error) {
  std::vector<std::string> log;
  const std::size_t num_threads = 10;
  tensorstore::execution::submit(
      tensorstore::MakeSyncFlowSender(
          ConcurrentSender{num_threads, /*.error=*/true}),
      tensorstore::LoggingReceiver{&log});
  ASSERT_EQ(num_threads + 3, log.size());
  EXPECT_EQ("set_starting", log[0]);
  EXPECT_EQ("set_error: 3", log[log.size() - 2]);
  EXPECT_EQ("set_stopping", log[log.size() - 1]);
  EXPECT_THAT(
      log, ::testing::UnorderedElementsAre(
               "set_starting", "set_value: 0", "set_value: 1", "set_value: 2",
               "set_value: 3", "set_value: 4", "set_value: 5", "set_value: 6",
               "set_value: 7", "set_value: 8", "set_value: 9", "set_error: 3",
               "set_stopping"));
}

}  // namespace
