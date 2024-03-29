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

#include "tensorstore/util/execution/collecting_sender.h"

#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/execution/sender_util.h"
#include "tensorstore/util/span.h"

namespace {

struct X {
  explicit X(int value) : value(value) {}

  int value;

  friend std::ostream& operator<<(std::ostream& os, const std::vector<X>& vec) {
    for (auto v : vec) {
      os << v.value << ' ';
    }
    return os;
  }
};

TEST(CollectingSenderTest, SuccessX) {
  std::vector<std::string> log;
  std::vector<int> input{1, 2, 3, 4};

  tensorstore::execution::submit(
      tensorstore::internal::MakeCollectingSender<std::vector<X>>(
          tensorstore::RangeFlowSender<tensorstore::span<int>>{input}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 1 2 3 4 "));
}

struct Y {
  explicit Y(int value) : value(value) {}

  int value;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Y& x) {
    absl::Format(&sink, "%d", x.value);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const std::vector<Y>& vec) {
    sink.Append(absl::StrJoin(vec, " "));
  }
};

TEST(CollectingSenderTest, SuccessY) {
  std::vector<std::string> log;
  std::vector<int> input{1, 2, 3, 4};

  tensorstore::execution::submit(
      tensorstore::internal::MakeCollectingSender<std::vector<Y>>(
          tensorstore::RangeFlowSender<tensorstore::span<int>>{input}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_value: 1 2 3 4"));
}

TEST(CollectingSenderTest, Error) {
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::internal::MakeCollectingSender<std::vector<X>>(
          tensorstore::FlowSingleSender<tensorstore::ErrorSender<int>>{5}),
      tensorstore::LoggingReceiver{&log});
  EXPECT_THAT(log, ::testing::ElementsAre("set_error: 5"));
}

}  // namespace
