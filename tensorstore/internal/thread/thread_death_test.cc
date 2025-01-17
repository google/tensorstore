// Copyright 2025 The TensorStore Authors
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

#if !defined(_WIN32) && !defined(__APPLE__) && defined(__has_include)
#if __has_include(<pthread.h>)
#define TENSORSTORE_INTERNAL_ENABLE_FORK_TEST 1
// In order to run the death test, both pthreads and fork are required,
// and the test must be run in gunit's thread-safe mode.
#endif
#endif

#if defined(TENSORSTORE_INTERNAL_ENABLE_FORK_TEST)

#include <pthread.h>
#include <unistd.h>

#include <gtest/gtest.h>
#include "tensorstore/internal/thread/thread.h"

namespace {

TEST(ThreadDeathTest, Fork) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // Span a thread and join it; this should register pthread_at_fork()
  // handler which will crash if fork() is called.
  int x = 0;
  tensorstore::internal::Thread my_thread({}, [&x]() { x = 1; });
  my_thread.Join();
  EXPECT_EQ(1, x);

  // fork()-ing after starting a thread is not supported; forcibly crash.
  EXPECT_DEATH(fork(), "");
}

}  // namespace

#endif  // TENSORSTORE_INTERNAL_ENABLE_FORK_TEST
