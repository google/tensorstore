// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/util/stop_token.h"

#include <functional>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/concurrent_testutil.h"

namespace {

TEST(StopTokenTest, Invariants) {
  tensorstore::StopSource source;
  EXPECT_TRUE(source.stop_possible());
  EXPECT_FALSE(source.stop_requested());

  tensorstore::StopToken token = source.get_token();
  EXPECT_TRUE(source.stop_possible());
  EXPECT_FALSE(source.stop_requested());

  EXPECT_EQ(token, source.get_token());

  EXPECT_TRUE(source.request_stop());

  EXPECT_TRUE(source.stop_possible());
  EXPECT_TRUE(source.stop_requested());
  EXPECT_TRUE(token.stop_requested());

  {
    tensorstore::StopSource source2;
    EXPECT_NE(token, source2.get_token());
  }
}

TEST(StopTokenTest, Invariants_Null) {
  tensorstore::StopSource source(nullptr);
  EXPECT_FALSE(source.stop_possible());
  EXPECT_FALSE(source.stop_requested());

  tensorstore::StopToken token = source.get_token();
  EXPECT_FALSE(source.stop_possible());
  EXPECT_FALSE(source.stop_requested());

  EXPECT_EQ(token, source.get_token());

  EXPECT_FALSE(source.request_stop());
  EXPECT_FALSE(source.stop_possible());
  EXPECT_FALSE(source.stop_requested());
  EXPECT_FALSE(token.stop_requested());

  {
    tensorstore::StopSource source2;
    EXPECT_NE(token, source2.get_token());
  }
}

TEST(StopTokenTest, Basic_InScope) {
  tensorstore::StopSource source;
  bool called = false;

  {
    tensorstore::StopCallback callback(source.get_token(),
                                       [&]() { called = true; });
    EXPECT_FALSE(called);
    EXPECT_TRUE(source.request_stop());
  }

  EXPECT_TRUE(called);
}

TEST(StopTokenTest, Basic_NotInScope) {
  tensorstore::StopSource source;

  bool called = false;
  {
    tensorstore::StopCallback callback(source.get_token(),
                                       [&]() { called = true; });
    EXPECT_FALSE(called);
  }

  EXPECT_TRUE(source.request_stop());
  EXPECT_FALSE(called);
}

TEST(StopTokenTest, Basic_Null) {
  tensorstore::StopSource source(nullptr);
  bool called = false;

  {
    tensorstore::StopCallback callback(source.get_token(),
                                       [&]() { called = true; });
    EXPECT_FALSE(called);
    EXPECT_FALSE(source.request_stop());
  }

  EXPECT_FALSE(called);
}

TEST(StopTokenTest, StopAlreadyRequested) {
  tensorstore::StopSource source;
  EXPECT_TRUE(source.request_stop());

  bool called = false;
  tensorstore::StopCallback callback(source.get_token(),
                                     [&]() { called = true; });
  EXPECT_TRUE(called);
}

TEST(StopTokenTest, CallbackOrder) {
  bool called[3] = {};

  auto do_nothing = []() {};
  using DoNothingCallback = tensorstore::StopCallback<decltype(do_nothing)>;

  /// Callbacks are invoked in reverse-order
  tensorstore::StopSource source;
  auto x = std::make_unique<DoNothingCallback>(source.get_token(), do_nothing);

  tensorstore::StopCallback callback0(source.get_token(), [&]() {
    EXPECT_TRUE(called[1]);
    called[0] = true;
  });
  tensorstore::StopCallback callback1(source.get_token(), [&]() {
    EXPECT_TRUE(called[2]);
    called[1] = true;
  });
  tensorstore::StopCallback callback2(source.get_token(), [&]() {
    EXPECT_FALSE(called[0]);
    called[2] = true;
  });

  /// Remove callbacks to mutate the list. Removing the first one (x)
  /// ensures that the list is maintained in the correct order.
  { DoNothingCallback tmp(source.get_token(), do_nothing); }
  x = nullptr;  // remove the first registered callback

  EXPECT_TRUE(source.request_stop());
  EXPECT_TRUE(called[2]);
}

// Tests that the callback is invoked with the appropriate value category
// (i.e. lvalue or rvalue).
TEST(StopCallbackTest, InvokeValueCategory) {
  struct Callback {
    void operator()() const& { value += 1; }
    void operator()() && { value += 100; }
    int& value;
  };

  tensorstore::StopSource source;

  int counts[3] = {};
  tensorstore::StopCallback stop_callback0(source.get_token(),
                                           Callback{counts[0]});
  Callback callback1{counts[1]};
  tensorstore::StopCallback<Callback&> stop_callback1(source.get_token(),
                                                      callback1);
  tensorstore::StopCallback<const Callback> stop_callback2(source.get_token(),
                                                           Callback{counts[2]});

  source.request_stop();

  EXPECT_THAT(counts, ::testing::ElementsAre(100, 1, 1));
}

TEST(StopTokenTest, SelfDeregister) {
  tensorstore::StopSource source;

  std::optional<tensorstore::StopCallback<std::function<void()>>> callback{
      std::in_place, source.get_token(), [&] { callback = std::nullopt; }};

  EXPECT_TRUE(source.request_stop());
  EXPECT_FALSE(callback.has_value());
}

TEST(StopTokenTest, Concurrent) {
  tensorstore::StopSource source;
  bool called = false;

  std::optional<tensorstore::StopCallback<std::function<void()>>> callback;

  tensorstore::internal::TestConcurrent(
      /*num_iterations=*/100,
      /*initialize=*/
      [&] {
        tensorstore::StopSource new_source;
        source = std::move(new_source);
        called = false;
      },
      /*finalize=*/
      [&] {
        EXPECT_TRUE(source.stop_requested());
        callback = std::nullopt;
        EXPECT_TRUE(called);
      },  //
      // Concurrently:
      // (a) register a callback.
      [&] { callback.emplace(source.get_token(), [&]() { called = true; }); },
      // (b) request a stop.
      [&] { source.request_stop(); },
      // (c) register and unregister a do-nothing callback.
      [&] {
        tensorstore::StopCallback callback(source.get_token(), []() {});
      } /**/
  );
}

}  // namespace
