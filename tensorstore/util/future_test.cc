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

/// Tests of Future and Promise.

#include "tensorstore/util/future.h"

#include <atomic>
#include <chrono>  // NOLINT
#include <functional>
#include <memory>
#include <thread>  // NOLINT
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future_impl.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {
using tensorstore::Future;
using tensorstore::FutureCallbackRegistration;
using tensorstore::InlineExecutor;
using tensorstore::IsFutureConvertible;
using tensorstore::MakeReadyFuture;
using tensorstore::MakeResult;
using tensorstore::MatchesStatus;
using tensorstore::Promise;
using tensorstore::PromiseFuturePair;
using tensorstore::ReadyFuture;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::internal::TestConcurrent;
using tensorstore::internal_future::FutureAccess;

static_assert(IsFutureConvertible<int, const int>::value, "");
static_assert(!IsFutureConvertible<const int, int>::value, "");
static_assert(
    std::is_same<
        decltype(FutureAccess::rep_pointer(std::declval<Future<void>&>())),
        tensorstore::internal_future::FutureStatePointer&>::value,
    "");
static_assert(
    std::is_same<
        decltype(
            FutureAccess::rep_pointer(std::declval<const Future<void>&>())),
        const tensorstore::internal_future::FutureStatePointer&>::value,
    "");
static_assert(
    std::is_same<
        decltype(FutureAccess::rep_pointer(std::declval<Future<void>&&>())),
        tensorstore::internal_future::FutureStatePointer&&>::value,
    "");
static_assert(
    std::is_same<
        decltype(FutureAccess::rep_pointer(std::declval<Promise<void>&>())),
        tensorstore::internal_future::PromiseStatePointer&>::value,
    "");
static_assert(
    std::is_same<
        decltype(
            FutureAccess::rep_pointer(std::declval<const Promise<void>&>())),
        const tensorstore::internal_future::PromiseStatePointer&>::value,
    "");
static_assert(
    std::is_same<
        decltype(FutureAccess::rep_pointer(std::declval<Promise<void>&&>())),
        tensorstore::internal_future::PromiseStatePointer&&>::value,
    "");

TEST(FutureTest, Valid) {
  EXPECT_FALSE(Future<int>().valid());
  EXPECT_FALSE(Promise<int>().valid());
  auto pair = PromiseFuturePair<int>::Make();
  EXPECT_TRUE(pair.future.valid());
  EXPECT_TRUE(pair.promise.valid());
  auto future2 = pair.promise.future();
  EXPECT_TRUE(future2.valid());
}

TEST(FutureTest, MakeReadyFuture) {
  Future<int> future = MakeReadyFuture<int>(3);
  EXPECT_EQ(true, future.ready());
  EXPECT_EQ(3, future.result().value());
  Result<int> result{tensorstore::in_place};
  bool got_result = false;
  future.ExecuteWhenReady([&](ReadyFuture<int> r) {
    got_result = true;
    result = r.result();
  });
  EXPECT_TRUE(got_result);
  EXPECT_EQ(result, future.result());
}

TEST(FutureTest, MakeInPlace) {
  auto pair = PromiseFuturePair<int>::Make(tensorstore::in_place, 4);
  pair.promise.reset();  // drop link.
  EXPECT_EQ(4, pair.future.value());
}

/// Tests that a ready future can be constructed implicitly.
TEST(FutureTest, ConstructFromValue) {
  Future<int> x = 3;
  EXPECT_EQ(3, x.value());
}

/// Tests that a ready `Future<const T>` can be constructed implicitly.
TEST(FutureTest, ConstructFromValueConst) {
  Future<const int> x = 3;
  EXPECT_EQ(3, x.value());
}

/// Tests that a `Result<Future<T>>` in an error state is implicitly flattened
/// to a ready `Future<T>` in an error state.
TEST(FutureTest, FlattenResultError) {
  Future<int> x = MakeResult<Future<int>>(absl::UnknownError("Error"));
  EXPECT_THAT(x.result(), MatchesStatus(absl::StatusCode::kUnknown, "Error"));
}

/// Tests that a `Result<Future<T>>` in an error state is implicitly flattened
/// to a ready `Future<const T>` in an error state.
TEST(FutureTest, FlattenResultErrorConst) {
  Future<const int> x = MakeResult<Future<int>>(absl::UnknownError("Error"));
  EXPECT_THAT(x.result(), MatchesStatus(absl::StatusCode::kUnknown, "Error"));
}

/// Tests that a `Result<Future<T>>` in an success state is implicitly flattened
/// to a `Future<T>`.
TEST(FutureTest, FlattenResultSuccess) {
  auto pair = PromiseFuturePair<int>::Make();
  Future<int> x = MakeResult(pair.future);
  EXPECT_TRUE(HaveSameSharedState(pair.future, x));
}

/// Same as above, but with conversion to Future<const int>.
TEST(FutureTest, FlattenResultSuccessConstConvert) {
  auto pair = PromiseFuturePair<int>::Make();
  Future<const int> x = MakeResult(pair.future);
  EXPECT_TRUE(HaveSameSharedState(pair.future, x));
}

/// Tests that a `Result<Future<int>>&` converts to `Future<int>`.
TEST(FutureTest, FlattenResultLvalue) {
  Result<Future<int>> f1 = absl::UnknownError("");
  Future<int> f2 = f1;
  EXPECT_EQ(absl::UnknownError(""), GetStatus(f2.result()));
}

/// Tests that Future::SetResult works.
TEST(FutureTest, SetResult) {
  auto pair = PromiseFuturePair<int>::Make();
  EXPECT_FALSE(pair.promise.ready());
  EXPECT_TRUE(pair.promise.result_needed());
  EXPECT_FALSE(pair.future.ready());
  Result<int> result{tensorstore::in_place};
  bool got_result = false;
  pair.future.ExecuteWhenReady([&](ReadyFuture<int> r) {
    got_result = true;
    result = r.result();
  });
  EXPECT_FALSE(got_result);
  EXPECT_TRUE(pair.promise.SetResult(5));
  EXPECT_FALSE(pair.promise.result_needed());
  EXPECT_TRUE(pair.future.ready());
  EXPECT_TRUE(pair.promise.ready());
  EXPECT_EQ(result, 5);
}

/// Tests that Future::Wait works.
TEST(FutureTest, Wait) {
  auto pair = PromiseFuturePair<int>::Make();
  std::thread thread(
      [](Promise<int> promise) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        EXPECT_TRUE(promise.SetResult(5));
      },
      std::move(pair.promise));
  pair.future.Wait();
  EXPECT_EQ(5, pair.future.result());
  thread.join();
}

TEST(FutureTest, WaitForFailure) {
  // This test is inherently flaky since it is timing-dependent.  Therefore, we
  // allow it to pass if it works even once.
  for (size_t i = 0; i < 100; ++i) {
    auto pair = PromiseFuturePair<int>::Make();
    EXPECT_FALSE(
        pair.future.WaitFor(absl::FromChrono(std::chrono::milliseconds(10))));
    std::thread thread(
        [](Promise<int> promise) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          EXPECT_TRUE(promise.SetResult(5));
        },
        pair.promise);
    const bool ready =
        pair.future.WaitFor(absl::FromChrono(std::chrono::milliseconds(5)));
    thread.join();
    if (!ready) {
      // Passed
      return;
    }
  }
  FAIL();
}

TEST(FutureTest, WaitForSuccess) {
  // This test is inherently flaky since it is timing-dependent.  Therefore, we
  // allow it to pass if it works even once.
  for (size_t i = 0; i < 100; ++i) {
    auto pair = PromiseFuturePair<int>::Make();
    std::thread thread(
        [](Promise<int> promise) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          EXPECT_TRUE(promise.SetResult(5));
        },
        pair.promise);
    const bool ready1 =
        pair.future.WaitFor(absl::FromChrono(std::chrono::milliseconds(20)));
    const bool ready2 =
        pair.future.WaitFor(absl::FromChrono(std::chrono::milliseconds(10)));
    thread.join();
    if (ready1 && ready2) {
      // Passed
      return;
    }
  }
  FAIL();
}

TEST(FutureTest, WaitUntilFailure) {
  // This test is inherently flaky since it is timing-dependent.  Therefore, we
  // allow it to pass if it works even once.
  for (size_t i = 0; i < 100; ++i) {
    auto pair = PromiseFuturePair<int>::Make();
    EXPECT_FALSE(pair.future.WaitUntil(absl::Now() - absl::Milliseconds(10)));
    EXPECT_FALSE(pair.future.WaitUntil(absl::Now() + absl::Milliseconds(10)));
    std::thread thread(
        [](Promise<int> promise) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          EXPECT_TRUE(promise.SetResult(5));
        },
        pair.promise);
    const bool ready =
        pair.future.WaitUntil(absl::Now() + absl::Milliseconds(5));
    thread.join();
    if (!ready) {
      // Passed
      return;
    }
  }
  FAIL();
}

TEST(FutureTest, WaitUntilSuccess) {
  // This test is inherently flaky since it is timing-dependent.  Therefore, we
  // allow it to pass if it works even once.
  for (size_t i = 0; i < 100; ++i) {
    auto pair = PromiseFuturePair<int>::Make();
    std::thread thread(
        [](Promise<int> promise) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          EXPECT_TRUE(promise.SetResult(5));
        },
        pair.promise);
    const bool ready1 =
        pair.future.WaitUntil(absl::Now() + absl::Milliseconds(20));
    const bool ready2 =
        pair.future.WaitUntil(absl::Now() + absl::Milliseconds(10));
    thread.join();
    if (ready1 && ready2) {
      // Passed
      return;
    }
  }
  FAIL();
}

/// Sets that SetResult can be called more than once.
TEST(FutureTest, SetResultTwice) {
  auto pair = PromiseFuturePair<int>::Make();
  EXPECT_TRUE(pair.promise.SetResult(3));
  EXPECT_EQ(3, pair.future.result());
  EXPECT_EQ(false, pair.promise.SetResult(5));
  EXPECT_EQ(3, pair.future.result());
}

/// Tests that basic usage of ExecuteWhenNotNeeded works.
TEST(FutureTest, ExecuteWhenNotNeeded) {
  auto pair = PromiseFuturePair<int>::Make();
  bool no_future = false;
  pair.promise.ExecuteWhenNotNeeded([&] { no_future = true; });
  EXPECT_FALSE(no_future);
  pair.future.reset();
  EXPECT_FALSE(pair.promise.result_needed());
  EXPECT_TRUE(no_future);
}

/// Tests that basic usage of ExecuteWhenNotNeeded works even after Force is
/// called.
TEST(FutureTest, ExecuteWhenNotNeededBeforeForced) {
  auto pair = PromiseFuturePair<int>::Make();
  bool no_future = false;
  pair.promise.ExecuteWhenNotNeeded([&] { no_future = true; });
  EXPECT_FALSE(no_future);
  pair.future.reset();
  EXPECT_FALSE(pair.promise.result_needed());
  EXPECT_TRUE(no_future);
}

/// Tests that unregistering an ExecuteWhenNotNeeded callback works.
TEST(FutureTest, ExecuteWhenNotNeededUnregister) {
  auto pair = PromiseFuturePair<int>::Make();
  bool no_future = false;
  auto registration =
      pair.promise.ExecuteWhenNotNeeded([&] { no_future = true; });
  EXPECT_FALSE(no_future);
  registration.Unregister();
  pair.future.reset();
  EXPECT_FALSE(no_future);
}

/// Tests that ExecuteWhenNotNeeded works when result_needed() == false.
TEST(FutureTest, ExecuteWhenNotNeededImmediate) {
  auto pair = PromiseFuturePair<int>::Make();
  bool no_future = false;
  pair.future.reset();
  auto registration =
      pair.promise.ExecuteWhenNotNeeded([&] { no_future = true; });
  EXPECT_TRUE(no_future);
  registration.Unregister();
}

/// Tests that a callback can be unregistered more than once.
TEST(FutureTest, ExecuteWhenReadyUnregisterTwice) {
  auto pair = PromiseFuturePair<int>::Make();
  bool invoked = false;
  auto registration =
      pair.future.ExecuteWhenReady([&](ReadyFuture<int>) { invoked = true; });
  EXPECT_FALSE(invoked);
  auto registration2 = registration;
  registration.Unregister();
  registration2.Unregister();
  pair.promise.SetResult(3);
  EXPECT_FALSE(invoked);
}

TEST(FutureTest, ExecuteWhenNotNeededThenForce) {
  auto pair = PromiseFuturePair<int>::Make();
  bool no_future = false;
  auto registration =
      pair.promise.ExecuteWhenNotNeeded([&] { no_future = true; });
  pair.future.Force();
  pair.future.reset();
  EXPECT_TRUE(no_future);
  registration.Unregister();
}

/// Tests that a callback can unregister itself.
TEST(FutureTest, ExecuteWhenReadyUnregisterSelf) {
  auto pair = PromiseFuturePair<int>::Make();
  bool invoked = false;
  FutureCallbackRegistration registration;
  registration = pair.future.ExecuteWhenReady([&](ReadyFuture<int>) {
    invoked = true;
    registration();
  });
  pair.promise.SetResult(3);
  EXPECT_TRUE(invoked);
}

/// Tests that a callback can unregister itself multiple times.
TEST(FutureTest, ExecuteWhenReadyUnregisterSelfTwice) {
  auto pair = PromiseFuturePair<int>::Make();
  bool invoked = false;
  FutureCallbackRegistration registration;
  registration = pair.future.ExecuteWhenReady([&](ReadyFuture<int>) {
    invoked = true;
    auto registration_copy = registration;
    registration();
    registration_copy();
  });
  pair.promise.SetResult(3);
  EXPECT_TRUE(invoked);
}

/// Tests that memory is deallocated (via address sanitizer).
TEST(FutureTest, Destructor) {
  auto pair = PromiseFuturePair<int>::Make();
  static_cast<void>(pair);
}

/// Tests that memory is deallocated (via address sanitizer).
TEST(FutureTest, DestructorExecuteWhenReady) {
  auto pair = PromiseFuturePair<int>::Make();
  pair.future.ExecuteWhenReady([&](ReadyFuture<int>) {});
}

/// Tests that a callback can unregister a later callback.
TEST(FutureTest, ExecuteWhenReadyUnregisterOther) {
  auto pair = PromiseFuturePair<int>::Make();
  bool invoked = false;
  FutureCallbackRegistration registration;
  pair.future.ExecuteWhenReady([&](ReadyFuture<int>) { registration(); });
  registration =
      pair.future.ExecuteWhenReady([&](ReadyFuture<int>) { invoked = true; });
  pair.promise.SetResult(3);
  EXPECT_FALSE(invoked);
}

/// Tests unregistering a ready callback while it is being invoked concurrently
/// from another thread.
TEST(FutureTest, ExecuteWhenReadyUnregisterConcurrent) {
  PromiseFuturePair<int> pair;
  std::atomic<bool> unregistered;
  FutureCallbackRegistration registration;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        unregistered = false;
        pair = PromiseFuturePair<int>::Make();
        registration = pair.future.ExecuteWhenReady([&](ReadyFuture<int>) {
          // Test that `unregistered` is not set to `true` before the
          // callback has finished running.
          for (int i = 0; i < 100; ++i) {
            EXPECT_FALSE(unregistered.load());
          }
        });
      },
      /*finalize=*/[&] { EXPECT_TRUE(unregistered.load()); },  //
      // Concurrently:
      // (a) mark pair.future ready (which invokes the callback)
      [&] { pair.promise.SetResult(3); },
      // (b) unregister the callback.
      [&] {
        registration.Unregister();
        unregistered = true;
      });
}

TEST(FutureTest, ExecuteWhenReadyUnregisterNonBlockingConcurrent) {
  PromiseFuturePair<int> pair;
  std::atomic<bool> callback_started, unregister_returned, callback_finished;
  FutureCallbackRegistration registration;
  TestConcurrent(
      /*num_iterations=*/1,
      /*initialize=*/
      [&] {
        callback_started = false;
        callback_finished = false;
        unregister_returned = false;
        pair = PromiseFuturePair<int>::Make();
        registration = pair.future.ExecuteWhenReady([&](ReadyFuture<int>) {
          callback_started = true;
          while (unregister_returned == false) {
            // spin wait
          }
          callback_finished = true;
        });
      },
      /*finalize=*/
      [&] {
        EXPECT_TRUE(callback_started);
        EXPECT_TRUE(unregister_returned);
        EXPECT_TRUE(callback_finished);
      },  //
      // Concurrently:
      // (a) mark pair.future ready (which invokes the callback)
      [&] { pair.promise.SetResult(3); },
      // (b) unregister the callback.
      [&] {
        while (!callback_started) {
          // spin wait
        }
        EXPECT_FALSE(callback_finished);
        registration.UnregisterNonBlocking();
        unregister_returned = true;
      });
}

/// Tests unregistering a result-not-needed callback while it is being invoked
/// concurrently from another thread.
TEST(FutureTest, ExecuteWhenNotNeededUnregisterConcurrent) {
  PromiseFuturePair<int> pair;
  std::atomic<bool> unregistered;
  FutureCallbackRegistration registration;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        unregistered = false;
        pair = PromiseFuturePair<int>::Make();
        registration = pair.promise.ExecuteWhenNotNeeded([&] {
          // Test that `unregistered` is not set to `true` before the
          // callback has finished running.
          for (int i = 0; i < 100; ++i) {
            EXPECT_FALSE(unregistered.load());
          }
        });
      },
      /*finalize=*/[&] { EXPECT_TRUE(unregistered.load()); },  //
      // Concurrently:
      // (a) mark pair.promise's result not needed (which invokes the callback)
      [&] { pair.promise.SetResult(3); },
      // (b) unregister the callback.
      [&] {
        registration.Unregister();
        unregistered = true;
      });
}

/// Tests unregistering a force callback while it is being invoked concurrently
/// from another thread.
TEST(FutureTest, ExecuteWhenForcedUnregisterConcurrent) {
  PromiseFuturePair<int> pair;
  std::atomic<bool> unregistered;
  FutureCallbackRegistration registration;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        unregistered = false;
        pair = PromiseFuturePair<int>::Make();
        registration = pair.promise.ExecuteWhenForced([&](Promise<int>) {
          // Test that `unregistered` is not set to `true` before the
          // callback has finished running.
          for (int i = 0; i < 100; ++i) {
            EXPECT_FALSE(unregistered.load());
          }
        });
      },
      /*finalize=*/[&] { EXPECT_TRUE(unregistered.load()); },  //
      // Concurrently:
      // (a) Force future (which invokes the callback)
      [&] { pair.future.Force(); },
      // (b) unregister the callback.
      [&] {
        registration.Unregister();
        unregistered = true;
      });
}

TEST(FutureTest, SetResultInForceCallback) {
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenForced([](Promise<int> p) { p.SetResult(5); });
  EXPECT_FALSE(pair.future.ready());
  pair.future.Force();
  EXPECT_EQ(true, pair.future.ready());
  EXPECT_EQ(5, pair.future.result());
}

TEST(FutureTest, ForceCallbackAddedAfterForced) {
  auto pair = PromiseFuturePair<int>::Make();
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  pair.future.Force();
  bool callback_ran = false;
  // Callback should be called and destroyed immediately.
  pair.promise.ExecuteWhenForced(
      [sentinel, &callback_ran](Promise<int> p) { callback_ran = true; });
  EXPECT_TRUE(callback_ran);
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_FALSE(pair.future.ready());
}

TEST(FutureTest, ForceCallbackAddedAfterForcedWithNoFuturesRemaining) {
  auto pair = PromiseFuturePair<int>::Make();
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  pair.future.Force();
  pair.future.reset();
  bool callback_ran = false;
  pair.promise.ExecuteWhenForced(
      [sentinel, &callback_ran](Promise<int> p) { callback_ran = true; });
  // Callback should be destroyed immediately without being called.
  EXPECT_FALSE(callback_ran);
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_FALSE(pair.promise.result_needed());
}

TEST(FutureTest, ForceCallbackDestroyedAfterForce) {
  auto pair = PromiseFuturePair<int>::Make();
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  pair.promise.ExecuteWhenForced(
      [sentinel](Promise<int> p) { p.SetResult(5); });
  EXPECT_EQ(2, sentinel.use_count());
  EXPECT_FALSE(pair.future.ready());
  pair.future.Force();
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_EQ(true, pair.future.ready());
  EXPECT_EQ(5, pair.future.result());
}

TEST(FutureTest, ForceAfterReady) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  pair.promise.ExecuteWhenForced(
      [&forced, sentinel](Promise<int> p) { forced = true; });
  EXPECT_EQ(2, sentinel.use_count());
  pair.promise.SetResult(3);
  // Force callback should have been destroyed.
  EXPECT_FALSE(forced);
  EXPECT_EQ(1, sentinel.use_count());
  pair.future.Force();
  EXPECT_FALSE(forced);
}

TEST(FutureTest, ForceCallbacksDestroyedWhenNoFuturesRemain) {
  auto pair = PromiseFuturePair<int>::Make();
  bool forced = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  pair.promise.ExecuteWhenForced(
      [&forced, sentinel](Promise<int> p) { forced = true; });
  EXPECT_EQ(2, sentinel.use_count());
  pair.future.reset();
  // Force callback should have been destroyed.
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_FALSE(forced);
}

struct CallOnCopy {
  CallOnCopy(const CallOnCopy& x)
      : call_when_copied(x.call_when_copied),
        call_when_invoked(x.call_when_invoked) {
    call_when_copied();
  }
  CallOnCopy(std::function<void()> call_when_copied,
             std::function<void()> call_when_invoked)
      : call_when_copied(call_when_copied),
        call_when_invoked(call_when_invoked) {}
  template <typename... Arg>
  void operator()(Arg&&...) {
    call_when_invoked();
  }
  std::function<void()> call_when_copied, call_when_invoked;
};

/// Tests that ExecuteWhenReady handles the case of the state becoming ready
/// after the initial check but before adding the callback to the list.
TEST(FutureTest, SetReadyCalledConcurrentlyWithExecuteWhenReady) {
  bool was_called = false;
  auto pair = PromiseFuturePair<int>::Make();
  pair.future.ExecuteWhenReady(CallOnCopy{[&] { pair.promise.SetResult(5); },
                                          [&] { was_called = true; }});
  EXPECT_TRUE(was_called);
  EXPECT_EQ(5, pair.future.result().value());
}

/// Tests that ExecuteWhenForce handles the case of Force being called after the
/// initial check but before adding the callback to the list.
TEST(FutureTest, ForceCalledConcurrentlyWithExecuteWhenForced) {
  bool was_called = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenForced(CallOnCopy{
      [&] { pair.future.Force(); }, [&, sentinel] { was_called = true; }});
  EXPECT_TRUE(was_called);
  EXPECT_EQ(1, sentinel.use_count());
}

/// Tests that ExecuteWhenForce handles the case of Force being called after the
/// initial check but before adding the callback to the list, and of SetResult
/// then being called while the callback is executing.
TEST(FutureTest, ForceAndThenSetResultCalledConcurrentlyWithExecuteWhenForced) {
  bool was_called = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenForced(CallOnCopy{[&] { pair.future.Force(); },
                                            [&, sentinel] {
                                              was_called = true;
                                              pair.promise.SetResult(5);
                                            }});
  EXPECT_TRUE(was_called);
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_EQ(5, pair.future.result().value());
}

/// Tests that ExecuteWhenNotNeeded handles the case of the last future
/// reference being released after the initial check but before adding the
/// callback to the list.
TEST(FutureTest, LastFutureReleasedConcurrentlyWithExecuteWhenNotNeeded) {
  bool was_called = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenNotNeeded(CallOnCopy{
      [&] { pair.future.reset(); }, [&, sentinel] { was_called = true; }});
  EXPECT_TRUE(was_called);
  EXPECT_EQ(1, sentinel.use_count());
}

/// Tests that ExecuteWhenForced handles the case of the last future reference
/// being released after the initial check but before adding the callback to the
/// list.
TEST(FutureTest, LastFutureReleasedConcurrentlyWithExecuteWhenForced) {
  bool was_called = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenForced(CallOnCopy{
      [&] { pair.future.reset(); }, [&, sentinel] { was_called = true; }});
  EXPECT_FALSE(was_called);
  EXPECT_EQ(1, sentinel.use_count());
}

/// Tests that ExecuteWhenForced handles the case of the state becoming ready
/// after the initial check but before adding the callback to the list.
TEST(FutureTest, SetResultCalledConcurrentlyWithExecuteWhenForced) {
  bool was_called = false;
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise.ExecuteWhenForced(
      CallOnCopy{[&] { pair.promise.SetResult(5); },
                 [&, sentinel] { was_called = true; }});
  EXPECT_FALSE(was_called);
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_EQ(5, pair.future.result().value());
}

TEST(FutureTest, PromiseBroken) {
  auto pair = PromiseFuturePair<int>::Make();
  pair.promise = {};
  EXPECT_TRUE(pair.future.ready());
  EXPECT_FALSE(pair.future.result().has_value());
  EXPECT_EQ(absl::UnknownError(""), pair.future.result().status());
}

TEST(FutureTest, ConvertInt) {
  auto pair = PromiseFuturePair<int>::Make();
  Future<const int> f = pair.future;
  Promise<const int> p = pair.promise;
}

TEST(FutureTest, ConvertVoid) {
  auto pair = PromiseFuturePair<void>::Make();
  Future<const void> f = pair.future;
  Promise<const void> p = pair.promise;
  pair.promise.SetResult(tensorstore::MakeResult());
  f.value();
}

TEST(FutureTest, ConvertVoid2) {
  auto pair = PromiseFuturePair<void>::Make();
  Future<const void> f = pair.future;
  Promise<const void> p = pair.promise;
  pair.promise.SetResult(absl::in_place);
  f.value();
}

struct NonMovable {
  NonMovable(int value) : value(value) {}
  NonMovable(NonMovable const&) = delete;
  NonMovable(NonMovable&&) = delete;

  int value;
};

TEST(FutureTest, NonMovableTypeInitialize) {
  auto pair = PromiseFuturePair<NonMovable>::Make(3);
  pair.promise.SetReady();
  EXPECT_EQ(3, pair.future.value().value);
}

TEST(FutureTest, NonMovableTypeSetReady) {
  auto pair = PromiseFuturePair<NonMovable>::Make();
  pair.promise.raw_result().emplace(5);
  pair.promise.SetReady();
  EXPECT_EQ(5, pair.future.value().value);
}

TEST(HaveSameSharedStateTest, Invalid) {
  Future<int> fa, fb;
  Future<const int> cf;
  Promise<int> pa, pb;
  Promise<int> cp;
  EXPECT_TRUE(HaveSameSharedState(fa, fb));
  EXPECT_TRUE(HaveSameSharedState(fa, cf));
  EXPECT_TRUE(HaveSameSharedState(pa, pb));
  EXPECT_TRUE(HaveSameSharedState(pa, fa));
  EXPECT_TRUE(HaveSameSharedState(fa, pb));
  EXPECT_TRUE(HaveSameSharedState(pa, cf));
}

TEST(HaveSameSharedStateTest, Valid) {
  auto pair1 = PromiseFuturePair<void>::Make();
  auto pair2 = PromiseFuturePair<void>::Make();
  EXPECT_TRUE(HaveSameSharedState(pair1.future, pair1.future));
  EXPECT_TRUE(HaveSameSharedState(pair1.future, pair1.promise));
  EXPECT_TRUE(HaveSameSharedState(pair1.promise, pair1.future));
  EXPECT_TRUE(HaveSameSharedState(pair1.promise, pair1.promise));
  EXPECT_FALSE(HaveSameSharedState(pair1.promise, pair2.promise));
  EXPECT_FALSE(HaveSameSharedState(pair1.promise, pair2.future));
  EXPECT_FALSE(HaveSameSharedState(pair1.future, pair2.future));
  EXPECT_FALSE(HaveSameSharedState(pair1.future, pair2.promise));
}

TEST(AcquireFutureReferenceTest, ExistingFutureNotReady) {
  auto pair = PromiseFuturePair<void>::Make();
  auto future2 = pair.promise.future();
  EXPECT_TRUE(HaveSameSharedState(future2, pair.future));
}

TEST(AcquireFutureReferenceTest, ExistingFutureReady) {
  auto pair = PromiseFuturePair<void>::Make();
  pair.promise.SetReady();
  auto future2 = pair.promise.future();
  EXPECT_TRUE(HaveSameSharedState(future2, pair.future));
}

TEST(AcquireFutureReferenceTest, NoExistingFutureNotReady) {
  auto pair = PromiseFuturePair<void>::Make();
  pair.future.reset();
  auto future2 = pair.promise.future();
  EXPECT_FALSE(future2.valid());
}

TEST(AcquireFutureReferenceTest, NoExistingFutureReady) {
  auto pair = PromiseFuturePair<void>::Make();
  pair.future.reset();
  pair.promise.SetReady();
  auto future2 = pair.promise.future();
  EXPECT_TRUE(HaveSameSharedState(future2, pair.promise));
}

TEST(LinkTest, MultipleSimple) {
  auto a_pair = PromiseFuturePair<int>::Make();
  auto b_pair = PromiseFuturePair<int>::Make();
  auto c_pair = PromiseFuturePair<int>::Make();
  EXPECT_FALSE(a_pair.future.ready());
  EXPECT_FALSE(b_pair.future.ready());
  EXPECT_FALSE(c_pair.future.ready());
  Link(
      [](Promise<int> c, ReadyFuture<int> a, ReadyFuture<int> b) {
        c.SetResult(a.result().value() + b.result().value());
      },
      c_pair.promise, a_pair.future, b_pair.future);
  a_pair.promise.SetResult(5);
  // The link callback is not yet invoked, because `b_pair.future` is still not
  // ready.
  EXPECT_FALSE(b_pair.future.ready());
  EXPECT_FALSE(c_pair.future.ready());
  b_pair.promise.SetResult(3);
  ASSERT_TRUE(c_pair.future.ready());
  EXPECT_EQ(8, c_pair.future.result().value());
}

TEST(LinkTest, EmptyCallback) {
  auto a_pair = PromiseFuturePair<int>::Make();
  auto b_pair = PromiseFuturePair<int>::Make();
  struct Callback {
    void operator()(Promise<int> b, ReadyFuture<int> a) const {
      b.SetResult(a.result().value());
    }
  };
  Link(Callback{}, b_pair.promise, a_pair.future);
  EXPECT_FALSE(a_pair.future.ready());
  EXPECT_FALSE(b_pair.future.ready());
  a_pair.promise.SetResult(5);
  ASSERT_TRUE(b_pair.future.ready());
  EXPECT_EQ(5, b_pair.future.result().value());
}

TEST(LinkValueTest, MultipleSuccessError) {
  auto a_pair = PromiseFuturePair<int>::Make();
  auto b_pair = PromiseFuturePair<int>::Make();
  auto c_pair = PromiseFuturePair<int>::Make();
  LinkValue(
      [](Promise<int> c, ReadyFuture<int> a, ReadyFuture<int> b) {
        c.SetResult(a.result().value() + b.result().value());
      },
      c_pair.promise, a_pair.future, b_pair.future);
  a_pair.promise.SetResult(5);
  EXPECT_FALSE(c_pair.future.ready());
  // The link callback is not yet invoked, because `b_pair.future` is not yet
  // ready.
  b_pair.promise.SetResult(absl::InvalidArgumentError("Test error"));
  ASSERT_TRUE(c_pair.future.ready());
  EXPECT_THAT(c_pair.future.result().status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, "Test error"));
}

TEST(LinkValueTest, MultipleErrorSuccess) {
  auto a_pair = PromiseFuturePair<int>::Make();
  auto b_pair = PromiseFuturePair<int>::Make();
  auto c_pair = PromiseFuturePair<int>::Make();
  LinkValue(
      [](Promise<int> c, ReadyFuture<int> a, ReadyFuture<int> b) {
        c.SetResult(a.result().value() + b.result().value());
      },
      c_pair.promise, a_pair.future, b_pair.future);
  b_pair.promise.SetResult(absl::InvalidArgumentError("Test error"));
  // The link is cancelled because `b_pair.future` became ready with an error.
  ASSERT_TRUE(c_pair.future.ready());
  EXPECT_THAT(c_pair.future.result().status(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, "Test error"));
}

TEST(LinkErrorTest, ImmediateSuccess) {
  auto pair = PromiseFuturePair<int>::Make(3);
  LinkError(pair.promise, MakeReadyFuture<int>(1));
  EXPECT_FALSE(pair.future.ready());
  pair.promise.reset();
  ASSERT_TRUE(pair.future.ready());
  EXPECT_EQ(3, pair.future.value());
}

TEST(LinkErrorTest, ImmediateFailure) {
  auto pair = PromiseFuturePair<int>::Make(3);
  LinkError(pair.promise, MakeReadyFuture<int>(absl::UnknownError("Msg")));
  pair.promise.reset();
  EXPECT_EQ(absl::UnknownError("Msg"), pair.future.result().status());
}

TEST(LinkErrorTest, DelayedSuccess) {
  auto pair1 = PromiseFuturePair<int>::Make(3);
  auto pair2 = PromiseFuturePair<void>::Make();
  LinkError(pair1.promise, pair2.future);
  pair1.promise.reset();
  // The only remaining reference to `pair1.promise` is now owned by the link.
  EXPECT_FALSE(pair1.future.ready());
  pair2.promise.SetResult(tensorstore::MakeResult());
  // Marking `pair2.promise` ready in a success state causes the link to
  // complete successfully, releasing the last reference to `pair1.promise`,
  // which marks it ready.
  ASSERT_TRUE(pair1.future.ready());
  EXPECT_EQ(3, pair1.future.value());
}

TEST(LinkErrorTest, DelayedFailure) {
  auto pair1 = PromiseFuturePair<int>::Make(3);
  auto pair2 = PromiseFuturePair<void>::Make();
  LinkError(pair1.promise, pair2.future);
  EXPECT_FALSE(pair1.future.ready());
  pair2.promise.SetResult(absl::UnknownError("Msg"));
  // Marking `pair2.promise` ready with an error causes the link to propagate
  // the error to `pair1.promise`.
  ASSERT_TRUE(pair1.future.ready());
  EXPECT_EQ(absl::UnknownError("Msg"), pair1.future.result().status());
}

TEST(LinkTest, SetReadyInForce) {
  auto pair1 = PromiseFuturePair<int>::Make();
  // `pair1.future` becomes ready immediately when it is forced.
  pair1.promise.ExecuteWhenForced([](Promise<int> self) { self.SetResult(5); });
  // `pair2.future` resolves to `pair1.future.value() + 2`.
  auto pair2 = PromiseFuturePair<int>::Make();
  Link([](Promise<int> p,
          ReadyFuture<int> f) { p.SetResult(f.result().value() + 2); },
       pair2.promise, pair1.future);
  // Initially neither future has been forced.
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.future.ready());
  // `pair2.future.result()` calls `pair2.future.Force()`, which due to the Link
  // calls `pair1.future.Force()`, which due to the `ExecuteWhenForced` callback
  // makes `pair1.future` ready, which due to the Link makes `pair2.future`
  // ready.
  EXPECT_EQ(7, pair2.future.result().value());
}

TEST(LinkTest, LinkAfterForceCalledWhereFutureBecomesReadyWhenForced) {
  auto pair1 = PromiseFuturePair<int>::Make();
  auto pair2 = PromiseFuturePair<int>::Make();
  // `pair2.future` becomes ready immediately when it is forced.
  pair2.promise.ExecuteWhenForced([](Promise<int> self) { self.SetResult(5); });
  pair1.future.Force();
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.future.ready());
  // Since `pair1.future.Force()` was called, this invokes
  // `pair2.future.Force()`, which causes `pair2.future` to become ready, which
  // then causes `pair1.future` to become ready.
  Link([](Promise<int> p,
          ReadyFuture<int> f1) { p.SetResult(f1.result().value() + 2); },
       pair1.promise, pair2.future);
  EXPECT_TRUE(pair1.future.ready());
  EXPECT_TRUE(pair2.future.ready());
  EXPECT_EQ(7, pair1.future.result().value());
}

TEST(LinkTest, LinkAfterForceCalledWhereFutureDoesNotBecomeReadyWhenForced) {
  auto pair1 = PromiseFuturePair<int>::Make();
  auto pair2 = PromiseFuturePair<int>::Make();
  pair1.future.Force();
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.future.ready());
  Link([](Promise<int> p,
          ReadyFuture<int> f1) { p.SetResult(f1.result().value() + 2); },
       pair1.promise, pair2.future);
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.future.ready());
  pair2.promise.SetResult(5);
  // Marking `pair2.promise` ready invokes the link callback, which marks
  // `pair1.promise` ready.
  EXPECT_TRUE(pair1.future.ready());
  EXPECT_TRUE(pair2.future.ready());
  EXPECT_EQ(7, pair1.future.result().value());
}

TEST(LinkTest, Unregister) {
  auto pair1 = PromiseFuturePair<int>::Make();
  pair1.promise.ExecuteWhenForced([](Promise<int> p) { p.SetResult(5); });
  auto pair2 = PromiseFuturePair<int>::Make();
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto registration = Link(
      [sentinel](Promise<int> p, ReadyFuture<int> f) {
        p.SetResult(f.result().value() + 2);
      },
      pair2.promise, pair1.future);
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.future.ready());
  EXPECT_EQ(2, sentinel.use_count());
  // Unregisters the link, which destroys the callback (including the copy of
  // `sentinel`).
  registration();
  EXPECT_EQ(1, sentinel.use_count());
  pair1.future.Force();
  // `pair1.future.Force()` does not invoke the callback.
  EXPECT_FALSE(pair2.future.ready());
}

TEST(LinkTest, AlreadyReady) {
  auto future1 = MakeReadyFuture<int>(5);
  auto pair2 = PromiseFuturePair<int>::Make();
  Link([](Promise<int> p,
          ReadyFuture<int> f) { p.SetResult(f.result().value() + 2); },
       pair2.promise, future1);
  // The link callback executes immediately since `future1` is already ready.
  EXPECT_TRUE(pair2.future.ready());
  EXPECT_EQ(7, pair2.future.result().value());
}

TEST(LinkTest, NotNeeded) {
  auto pair1 = PromiseFuturePair<int>::Make();
  auto pair2 = PromiseFuturePair<int>::Make();
  pair2.future.reset();
  EXPECT_FALSE(pair2.promise.result_needed());
  // Copied into the callback so that the `use_count()` can be used to track
  // when the callback is destroyed.
  auto sentinel = std::make_shared<int>();
  auto registration = Link(
      [sentinel](Promise<int> p, ReadyFuture<int> f) {
        p.SetResult(f.result().value() + 2);
      },
      pair2.promise, pair1.future);
  // The link is never registered because `pair2.promise` is not needed.
  EXPECT_EQ(1, sentinel.use_count());
  EXPECT_FALSE(pair1.future.ready());
  EXPECT_FALSE(pair2.promise.ready());
}

TEST(LinkTest, ConcurrentSetReady) {
  PromiseFuturePair<int> pair1, pair2, pair3;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pair1 = PromiseFuturePair<int>::Make();
        pair2 = PromiseFuturePair<int>::Make();
        pair3 = PromiseFuturePair<int>::Make();
        Link([](Promise<int> p1, ReadyFuture<int> f2,
                ReadyFuture<int> f3) { p1.SetResult(f2.value() + f3.value()); },
             pair1.promise, pair2.future, pair3.future);
      },
      /*finalize=*/
      [&] {
        ASSERT_TRUE(pair1.future.ready());
        EXPECT_EQ(pair1.future.value(), 7);
      },
      // Concurrently:
      // (a) mark pair2.future ready;
      [&] { pair2.promise.SetResult(5); },
      // (b) mark pair3.future ready.
      [&] { pair3.promise.SetResult(2); });
}

TEST(LinkTest, ConcurrentLinkAndSetReady) {
  PromiseFuturePair<int> pair1, pair2, pair3;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pair1 = PromiseFuturePair<int>::Make();
        pair2 = PromiseFuturePair<int>::Make();
        pair3 = PromiseFuturePair<int>::Make();
      },
      /*finalize=*/
      [&] {
        ASSERT_TRUE(pair1.future.ready());
        EXPECT_EQ(pair1.future.value(), 7);
      },
      // Concurrently:
      // (a) create the link;
      [&] {
        Link([](Promise<int> p1, ReadyFuture<int> f2,
                ReadyFuture<int> f3) { p1.SetResult(f2.value() + f3.value()); },
             pair1.promise, pair2.future, pair3.future);
      },
      // (b) mark pair2.future ready;
      [&] { pair2.promise.SetResult(5); },
      // (c) mark pair3.future ready.
      [&] { pair3.promise.SetResult(2); });
}

TEST(LinkTest, ConcurrentUnregister) {
  PromiseFuturePair<int> pair1, pair2;
  FutureCallbackRegistration registration;
  std::atomic<bool> unregistered;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        unregistered = false;
        pair1 = PromiseFuturePair<int>::Make(1);
        pair2 = PromiseFuturePair<int>::Make();
        registration = Link(
            [&](Promise<int> p1, ReadyFuture<int> f2) {
              // Test that `unregistered` is not set to `true` before the
              // callback has finished running.
              for (int i = 0; i < 100; ++i) {
                EXPECT_FALSE(unregistered.load());
              }
            },
            pair1.promise, pair2.future);
      },
      /*finalize=*/[&] { EXPECT_TRUE(unregistered.load()); },  //
      // Concurrently:
      // (a) mark pair2.future ready (which invokes the callback)
      [&] { pair2.promise.SetResult(2); },
      // (b) unregister the callback.
      [&] {
        registration.Unregister();
        unregistered = true;
      });
}

// Tests that deadlock does not occur when `pair1 -> pair2 -> pair3` are linked
// and `pair3` is forced while `pair1` is marked ready.  This relies on
// `FutureLink` calling `Unregister` on the force callback with `block=false`.
TEST(LinkTest, ConcurrentForceAndSetReady) {
  PromiseFuturePair<int> pair1, pair2, pair3;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pair1 = PromiseFuturePair<int>::Make(1);
        pair2 = PromiseFuturePair<int>::Make();
        pair3 = PromiseFuturePair<int>::Make();
        Link(pair2.promise, pair1.future);
        Link(pair3.promise, pair2.future);
      },
      /*finalize=*/[&] {},  //
      // Concurrently:
      // (a) mark pair1.future ready
      [&] { pair1.promise.SetResult(2); },
      // (b) Force pair3.future
      [&] { pair3.future.Force(); });
}

// Tests that Link works with no futures.
TEST(LinkTest, NoFutures) {
  auto pair = PromiseFuturePair<int>::Make();

  Link([](Promise<int> promise) { promise.SetResult(5); }, pair.promise);
  ASSERT_TRUE(pair.future.ready());
  ASSERT_TRUE(pair.future.result());
  EXPECT_EQ(5, pair.future.value());
}

// Tests the no callback overload of `Link`.
TEST(LinkTest, NoCallback) {
  auto [promise, future] = PromiseFuturePair<int>::Make();
  promise.ExecuteWhenForced([](Promise<int> promise) { promise.SetResult(5); });

  // Test unlinking before the future becomes ready.
  {
    auto [linked_promise, linked_future] = PromiseFuturePair<int>::Make();
    auto link = Link(linked_promise, future);
    EXPECT_FALSE(linked_future.ready());
    link.Unregister();
    linked_future.Force();
    EXPECT_FALSE(linked_future.ready());
    EXPECT_FALSE(future.ready());
  }

  // Test forcing.
  {
    auto [linked_promise, linked_future] = PromiseFuturePair<int>::Make();
    auto link = Link(linked_promise, future);
    EXPECT_FALSE(linked_future.ready());
    linked_future.Force();
    ASSERT_TRUE(linked_future.ready());
    ASSERT_TRUE(future.ready());
    EXPECT_THAT(linked_future.result(), ::testing::Optional(5));
  }

  // Test linking after the future is ready.
  {
    auto [linked_promise, linked_future] = PromiseFuturePair<int>::Make();
    auto link = Link(linked_promise, future);
    ASSERT_TRUE(linked_future.ready());
    EXPECT_THAT(linked_future.result(), ::testing::Optional(5));
  }
}

// Tests that Forcing a linked promise A, while the linked future B is
// concurrently marked ready, doesn't deadlock.
TEST(LinkErrorTest, ConcurrentForceAndSetReady) {
  PromiseFuturePair<void> pairA, pairB;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pairA = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        pairB = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        LinkError(pairA.promise, pairB.future);
      },
      /*finalize=*/
      [&] {
        EXPECT_TRUE(pairB.future.ready());
        EXPECT_TRUE(pairB.future.result());
        EXPECT_FALSE(pairA.future.ready());
      },  //
      // Concurrently:
      // (a) Force pairA.future
      [&] { pairA.future.Force(); },
      // (b) Mark pairB.promise ready in a success state.
      [&] { pairB.promise.SetReady(); });
}

// Tests that concurrently setting futures B and C ready in an error state,
// while they are both linked via LinkError to promise A, doesn't deadlock.
TEST(LinkErrorTest, ConcurrentSetError) {
  PromiseFuturePair<void> pairA, pairB, pairC;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pairA = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        pairB = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        pairC = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        LinkError(pairA.promise, pairB.future, pairC.future);
      },
      /*finalize=*/
      [&] {
        EXPECT_TRUE(pairA.future.ready());
        EXPECT_TRUE(pairB.future.ready());
        EXPECT_TRUE(pairC.future.ready());
        EXPECT_FALSE(pairA.future.result());
        EXPECT_FALSE(pairB.future.result());
        EXPECT_FALSE(pairC.future.result());
      },  //
      // Concurrently:
      // (a) Mark pairB.promise ready in an error state.
      [&] { pairB.promise.SetResult(absl::UnknownError("")); },
      // (b) Mark pairC.promise ready in an error state.
      [&] { pairC.promise.SetResult(absl::UnknownError("")); });
}

// Tests that Forcing a linked promise A, while the linked future B is
// concurrently marked ready with an error (causing the linked promise A to also
// be marked ready), doesn't deadlock.
TEST(LinkErrorTest, ConcurrentForceAndSetError) {
  PromiseFuturePair<void> pairA, pairB;
  TestConcurrent(
      /*num_iterations=*/1000,
      /*initialize=*/
      [&] {
        pairA = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        pairB = PromiseFuturePair<void>::Make(tensorstore::MakeResult());
        LinkError(pairA.promise, pairB.future);
      },
      /*finalize=*/
      [&] {
        EXPECT_TRUE(pairB.future.ready());
        EXPECT_TRUE(pairA.future.ready());
        EXPECT_FALSE(pairB.future.result());
        EXPECT_FALSE(pairA.future.result());
      },  //
      // Concurrently:
      // (a) Force pairA.future
      [&] { pairA.future.Force(); },
      // (b) Mark pairB.promise ready with an error.
      [&] { pairB.promise.SetResult(absl::UnknownError("")); });
}

TEST(PromiseFuturePairTest, LinkImmediateSuccess) {
  auto future = PromiseFuturePair<int>::Link(
                    [](Promise<int> p, ReadyFuture<int> f) {
                      p.SetResult(f.value() + 1);
                    },
                    MakeReadyFuture<int>(1))
                    .future;
  EXPECT_EQ(2, future.value());
}

TEST(PromiseFuturePairTest, LinkImmediateFailure) {
  auto future =
      PromiseFuturePair<int>::Link(
          [](Promise<int> p, ReadyFuture<int> f) { p.SetResult(f.result()); },
          MakeReadyFuture<int>(absl::UnknownError("Fail")))
          .future;
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

TEST(PromiseFuturePairTest, LinkDeferredSuccess) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::Link(
                    [](Promise<int> p, ReadyFuture<int> f) {
                      p.SetResult(f.value() + 1);
                    },
                    pair.future)
                    .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(1);
  EXPECT_EQ(2, future.value());
}

TEST(PromiseFuturePairTest, LinkDeferredFailure) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future =
      PromiseFuturePair<int>::Link(
          [](Promise<int> p, ReadyFuture<int> f) { p.SetResult(f.result()); },
          pair.future)
          .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(absl::UnknownError("Fail"));
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

TEST(PromiseFuturePairTest, LinkResultInit) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::Link(
                    5, [](Promise<int> p, ReadyFuture<int> f) {}, pair.future)
                    .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(3);
  EXPECT_EQ(5, future.value());
}

TEST(PromiseFuturePairTest, LinkValueImmediateSuccess) {
  auto future = PromiseFuturePair<int>::LinkValue(
                    [](Promise<int> p, ReadyFuture<int> f) {
                      p.SetResult(f.value() + 1);
                    },
                    MakeReadyFuture<int>(1))
                    .future;
  EXPECT_EQ(2, future.value());
}

TEST(PromiseFuturePairTest, LinkValueImmediateFailure) {
  auto future = PromiseFuturePair<int>::LinkValue(
                    [](Promise<int> p, ReadyFuture<int> f) {},
                    MakeReadyFuture<int>(absl::UnknownError("Fail")))
                    .future;
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

TEST(PromiseFuturePairTest, LinkValueDeferredSuccess) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::LinkValue(
                    [](Promise<int> p, ReadyFuture<int> f) {
                      p.SetResult(f.value() + 1);
                    },
                    pair.future)
                    .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(1);
  EXPECT_EQ(2, future.value());
}

TEST(PromiseFuturePairTest, LinkValueDeferredFailure) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::LinkValue(
                    [](Promise<int> p, ReadyFuture<int> f) {}, pair.future)
                    .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(absl::UnknownError("Fail"));
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

TEST(PromiseFuturePairTest, LinkValueResultInit) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::LinkValue(
                    5, [](Promise<int> p, ReadyFuture<int> f) {}, pair.future)
                    .future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(3);
  EXPECT_EQ(5, future.value());
}

TEST(PromiseFuturePairTest, LinkErrorImmediateSuccess) {
  auto future =
      PromiseFuturePair<int>::LinkError(3, MakeReadyFuture<int>(1)).future;
  EXPECT_EQ(3, future.value());
}

TEST(PromiseFuturePairTest, LinkErrorImmediateFailure) {
  auto future = PromiseFuturePair<int>::LinkError(
                    3, MakeReadyFuture<int>(1),
                    MakeReadyFuture<int>(absl::UnknownError("Fail")))
                    .future;
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

TEST(PromiseFuturePairTest, LinkErrorDeferredSuccess) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future = PromiseFuturePair<int>::LinkError(3, pair.future).future;
  EXPECT_FALSE(future.ready());
  pair.promise.SetResult(5);
  EXPECT_EQ(3, future.value());
}

TEST(PromiseFuturePairTest, LinkErrorDeferredFailure) {
  auto pair = PromiseFuturePair<int>::Make();
  auto future =
      PromiseFuturePair<int>::LinkError(3, MakeReadyFuture<int>(1), pair.future)
          .future;
  EXPECT_FALSE(pair.future.ready());
  pair.promise.SetResult(absl::UnknownError("Fail"));
  EXPECT_THAT(future.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Fail"));
}

// Tests that the callback passed to `Link` is destroyed after it is called.
TEST(LinkTest, DestroyCallback) {
  auto pair1 = PromiseFuturePair<int>::Make();
  auto sentinel = std::make_shared<bool>(false);
  auto pair2 = PromiseFuturePair<void>::Make();
  auto registration =
      Link([sentinel](Promise<void>, ReadyFuture<int>) { *sentinel = true; },
           pair2.promise, pair1.future);
  EXPECT_EQ(2, sentinel.use_count());
  pair1.promise.SetResult(1);
  EXPECT_EQ(true, *sentinel);
  EXPECT_EQ(1, sentinel.use_count());
}

// Tests that the callback passed to `PromiseFuturePair::Link` is destroyed
// after it is called.
TEST(PromiseFuturePairTest, LinkDestroyCallback) {
  auto pair1 = PromiseFuturePair<int>::Make();
  auto sentinel = std::make_shared<bool>(false);
  auto pair2 = PromiseFuturePair<void>::Link(
      [sentinel](Promise<void>, ReadyFuture<int>) { *sentinel = true; },
      pair1.future);
  EXPECT_EQ(2, sentinel.use_count());
  pair1.promise.SetResult(1);
  EXPECT_EQ(true, *sentinel);
  EXPECT_EQ(1, sentinel.use_count());
}

// Tests that MapFuture works with no futures.
TEST(MapFutureTest, NoFutures) {
  auto future = MapFuture(InlineExecutor{}, [] { return 3; });
  ASSERT_TRUE(future.ready());
  ASSERT_TRUE(future.result());
  EXPECT_EQ(3, future.value());
}

TEST(MapFutureTest, BothReady) {
  auto a = MakeReadyFuture<int>(3);
  auto b = MakeReadyFuture<int>(5);
  auto c = MapFuture(
      InlineExecutor{},
      [](Result<int> a, Result<int> b) -> Result<int> {
        return MapResult(std::plus<int>{}, a, b);
      },
      a, b);
  EXPECT_EQ(8, c.result().value());
}

TEST(MapFutureTest, NonConstOperator) {
  struct MyStruct {
    Result<int> operator()() { return 2; }
  };

  Future<int> x = MapFuture(InlineExecutor{}, MyStruct{});
  EXPECT_EQ(2, x.result().value());
}

TEST(MapFutureTest, LValueReference) {
  auto a = MakeReadyFuture<int>(3);
  EXPECT_EQ(3, a.value());
  auto b = MapFuture(
      InlineExecutor{},
      [&](Result<int>& value) {
        value = 10;
        return 7;
      },
      a);
  EXPECT_EQ(7, b.value());
  EXPECT_EQ(10, a.value());
}

TEST(MapFutureValueTest, BothReady) {
  auto a = MakeReadyFuture<int>(3);
  auto b = MakeReadyFuture<int>(5);
  auto c = MapFutureValue(InlineExecutor{}, std::plus<int>{}, a, b);
  EXPECT_EQ(8, c.result().value());
}

TEST(MapFutureValueTest, LValueReference) {
  auto a = MakeReadyFuture<int>(3);
  EXPECT_EQ(3, a.value());
  auto b = MapFutureValue(
      InlineExecutor{},
      [&](int& value) {
        value = 10;
        return 7;
      },
      a);
  EXPECT_EQ(7, b.value());
  EXPECT_EQ(10, a.value());
}

TEST(MapFutureValueTest, ValueToError) {
  auto a = MakeReadyFuture<int>(3);
  auto b = MapFutureValue(
      InlineExecutor{},
      [](int x) -> Result<int> {
        return absl::UnknownError(tensorstore::StrCat("Got value: ", x));
      },
      a);
  EXPECT_THAT(b.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Got value: 3"));
}

TEST(MapFutureErrorTest, Success) {
  auto pair = PromiseFuturePair<int>::Make();
  auto mapped = MapFutureError(
      InlineExecutor{}, [](Status status) { return 5; }, pair.future);
  EXPECT_FALSE(mapped.ready());
  pair.promise.SetResult(7);
  EXPECT_EQ(7, mapped.result());
}

TEST(MapFutureErrorTest, ErrorMappedToSuccess) {
  auto pair = PromiseFuturePair<int>::Make();
  auto mapped = MapFutureError(
      InlineExecutor{},
      [](Status status) {
        EXPECT_EQ(absl::UnknownError("message"), status);
        return 5;
      },
      pair.future);
  EXPECT_FALSE(mapped.ready());
  pair.promise.SetResult(absl::UnknownError("message"));
  EXPECT_EQ(5, mapped.result());
}

TEST(MapFutureErrorTest, ErrorMappedToError) {
  auto pair = PromiseFuturePair<int>::Make();
  auto mapped = MapFutureError(
      InlineExecutor{},
      [](Status status) {
        return tensorstore::MaybeAnnotateStatus(status, "Mapped");
      },
      pair.future);
  EXPECT_FALSE(mapped.ready());
  pair.promise.SetResult(absl::UnknownError("message"));
  EXPECT_THAT(mapped.result(),
              MatchesStatus(absl::StatusCode::kUnknown, "Mapped: message"));
}

TEST(MakeReadyFutureTest, Basic) {
  auto future = MakeReadyFuture();
  static_assert(std::is_same_v<ReadyFuture<const void>, decltype(future)>);
  EXPECT_TRUE(future.ready());
  EXPECT_EQ(MakeResult(), future.result());
}

TEST(FutureTest, SetDeferredResult) {
  auto [promise, future] = PromiseFuturePair<int>::Make();
  SetDeferredResult(promise, 2);
  EXPECT_FALSE(future.ready());
  SetDeferredResult(promise, 3);
  EXPECT_FALSE(future.ready());
  promise = Promise<int>();
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), ::testing::Optional(2));
}

TEST(FutureTest, SetDeferredResultAfterReady) {
  auto [promise, future] = PromiseFuturePair<int>::Make();
  promise.SetResult(1);
  ASSERT_TRUE(future.ready());
  SetDeferredResult(promise, 2);
  ASSERT_TRUE(future.ready());
  EXPECT_THAT(future.result(), ::testing::Optional(1));
}

}  // namespace
