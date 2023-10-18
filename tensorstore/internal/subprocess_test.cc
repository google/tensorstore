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

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/env.h"
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
  ASSERT_TRUE(child.ok());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);
}

TEST(SubprocessTest, Kill) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSleepArg, kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  ASSERT_TRUE(child.ok());

  EXPECT_THAT(child->Join(/*block=*/false),
              tensorstore::MatchesStatus(absl::StatusCode::kUnavailable));

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
  ASSERT_TRUE(child.ok());

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
  ASSERT_TRUE(child.ok());

  // Kill changes the result;
  child->Kill().IgnoreError();
}

TEST(SubprocessTest, Env) {
  /// Should be able to spawn and just discard the process.
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {"--env=SUBPROCESS_TEST_ENV"};
  opts.env = absl::flat_hash_map<std::string, std::string>(
      {{"SUBPROCESS_TEST_ENV", "1"}});

  auto child = SpawnSubprocess(opts);
  ASSERT_TRUE(child.ok());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 41);
}

}  // namespace

int main(int argc, char* argv[]) {
  program_name = new std::string(argv[0]);
  ABSL_LOG(INFO) << *program_name;

  for (int i = 1; i < argc; i++) {
    std::string_view argv_i(argv[i]);
    if (argv_i == kSubprocessArg) {
      printf("PASS\n");
      return 33;
    }
    if (argv_i == kSleepArg) {
      absl::SleepFor(absl::Seconds(1));
    }
    if (absl::StartsWith(argv_i, "--env=")) {
      auto env_str = argv_i.substr(6);
      if (env_str.empty()) {
        return 40;
      }
      if (tensorstore::internal::GetEnv(env_str.data()).has_value()) {
        return 41;
      }
      return 42;
    }
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
