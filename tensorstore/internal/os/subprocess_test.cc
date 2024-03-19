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

#include "tensorstore/internal/os/subprocess.h"

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
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::SubprocessOptions;

static std::string* program_name = nullptr;
const char kSubprocessArg[] = "--is_subprocess";
const char kSleepArg[] = "--sleep";

TEST(SubprocessTest, Join) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};
  auto child = SpawnSubprocess(opts);
  TENSORSTORE_ASSERT_OK(child.status());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);
}

TEST(SubprocessTest, Kill) {
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSleepArg, kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  TENSORSTORE_ASSERT_OK(child.status());

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
  opts.stdout_action = SubprocessOptions::Ignore();
  opts.stderr_action = SubprocessOptions::Ignore();

  auto child = SpawnSubprocess(opts);
  TENSORSTORE_ASSERT_OK(child.status());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);
}

TEST(SubprocessTest, Redirects) {
  ::tensorstore::internal_testing::ScopedTemporaryDirectory temp_dir;
  std::string out_file = JoinPath(temp_dir.path(), "stdout");

  /// Should be able to spawn and just discard the process.
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};
  opts.env.emplace(::tensorstore::internal::GetEnvironmentMap());
  opts.env->insert_or_assign("SUBPROCESS_TEST_ENV", "1");
  opts.stdout_action = SubprocessOptions::Redirect{out_file};
  opts.stderr_action = SubprocessOptions::Redirect{out_file};

  auto child = SpawnSubprocess(opts);
  TENSORSTORE_ASSERT_OK(child.status());

  int exit_code;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(exit_code, child->Join());
  EXPECT_EQ(exit_code, 33);

  // Expect that the file exists.
  std::string filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(riegeli::FdReader(out_file), filedata));
  EXPECT_THAT(filedata, ::testing::HasSubstr("PASS"));
}

TEST(SubprocessTest, Drop) {
  /// Should be able to spawn and just discard the process.
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {kSubprocessArg};

  auto child = SpawnSubprocess(opts);
  TENSORSTORE_ASSERT_OK(child.status());

  // Kill changes the result;
  child->Kill().IgnoreError();
}

TEST(SubprocessTest, Env) {
  /// Should be able to spawn and just discard the process.
  SubprocessOptions opts;
  opts.executable = *program_name;
  opts.args = {"--env=SUBPROCESS_TEST_ENV"};
  opts.env = absl::flat_hash_map<std::string, std::string>({
#ifdef _WIN32
      {"PATH", ::tensorstore::internal::GetEnv("PATH").value_or("")},
#endif
      {"SUBPROCESS_TEST_ENV", "1"}});

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
