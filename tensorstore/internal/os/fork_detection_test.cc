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

#if !defined(_WIN32)
#define TENSORSTORE_INTERNAL_ENABLE_FORK_TEST 1
#endif

#if defined(TENSORSTORE_INTERNAL_ENABLE_FORK_TEST)

#include "tensorstore/internal/os/fork_detection.h"

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::tensorstore::internal_os::AbortIfForkDetected;
using ::tensorstore::internal_os::SetupForkDetection;
using ::testing::Gt;

namespace {

int DoFork() {
  int status = 0;

  pid_t p = fork();
  if (p < 0) {
    return -1;
  } else if (p == 0) {
    // child process aborts here.
    AbortIfForkDetected();
  } else {
    // Wait for the child process to exit and return its status.
    waitpid(p, &status, 0);
  }
  return status;
}

TEST(ForkDetectionDeathTest, Fork) {
  // The test must be run in gunit's thread-safe mode.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  SetupForkDetection();

  EXPECT_THAT(DoFork(), Gt(0));
}

}  // namespace

#endif  // TENSORSTORE_INTERNAL_ENABLE_FORK_TEST
