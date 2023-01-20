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

#include "tensorstore/internal/subprocess.h"

#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;

static std::string* program_name = nullptr;
const char kSubprocessArg[] = "--is_subprocess";
const char kSleepArg[] = "--sleep";

TEST(SubprocessTest, Join) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  EXPECT_TRUE(child.ok());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);
}

TEST(SubprocessTest, Kill) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSleepArg, kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  EXPECT_TRUE(child.ok());

  child->Kill().IgnoreError();

  // exit code on killed process is os dependent.
  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_NE(exit_code, 33);
}

TEST(SubprocessTest, DontInherit) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};
  opts.inherit_stdout = false;
  opts.inherit_stderr = false;

  auto child = SpawnSubprocess(opts);
  EXPECT_TRUE(child.ok());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);
}

TEST(SubprocessTest, Drop) {
  /// Should be able to spawn and just discard the process.
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  EXPECT_TRUE(child.ok());

  // Kill changes the result;
  child->Kill().IgnoreError();
}

}  // namespace

int main(int argc, char* argv[]) {
  program_name = new std::string(argv[0]);
  ABSL_LOG(INFO) << *program_name;

  for (int i = 1; i < argc; i++) {
    if (std::string_view(argv[i]) == kSubprocessArg) {
      printf("PASS\n");
      return 33;
    }
    if (std::string_view(argv[i]) == kSleepArg) {
      absl::SleepFor(absl::Seconds(1));
    }
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
