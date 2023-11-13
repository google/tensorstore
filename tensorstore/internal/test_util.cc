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

#include "tensorstore/internal/test_util.h"

#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/strings/numbers.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/os/filesystem.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/kvstore/file/file_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

ScopedTemporaryDirectory::ScopedTemporaryDirectory() {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";

  absl::BitGen gen;
  char data[24];
  for (auto& x : data) {
    x = kAlphabet[absl::Uniform(gen, 0u, std::size(kAlphabet) - 1)];
  }

  std::string basename = tensorstore::StrCat(
      "tmp_tensorstore_test_", std::string_view(data, std::size(data)));
  path_ = JoinPath(internal_os::TemporaryDirectoryPath(), basename);

  TENSORSTORE_CHECK_OK(internal_os::MakeDirectory(path_));
}

ScopedTemporaryDirectory::~ScopedTemporaryDirectory() {
  TENSORSTORE_CHECK_OK(internal_os::RemoveAll(path_));
}

ScopedCurrentWorkingDirectory::ScopedCurrentWorkingDirectory(
    const std::string& new_cwd) {
  TENSORSTORE_CHECK_OK_AND_ASSIGN(old_cwd_, internal_file_util::GetCwd());
  TENSORSTORE_CHECK_OK(internal_file_util::SetCwd(new_cwd));
}

ScopedCurrentWorkingDirectory::~ScopedCurrentWorkingDirectory() {
  TENSORSTORE_CHECK_OK(internal_file_util::SetCwd(old_cwd_));
}

void RegisterGoogleTestCaseDynamically(std::string test_suite_name,
                                       std::string test_name,
                                       std::function<void()> test_func,
                                       SourceLocation loc) {
  struct Fixture : public ::testing::Test {};
  class Test : public Fixture {
   public:
    Test(const std::function<void()>& test_func) : test_func_(test_func) {}
    void TestBody() override { test_func_(); }

   private:
    std::function<void()> test_func_;
  };
  ::testing::RegisterTest(test_suite_name.c_str(), test_name.c_str(),
                          /*type_param=*/nullptr,
                          /*value_param=*/nullptr, loc.file_name(), loc.line(),
                          [test_func = std::move(test_func)]() -> Fixture* {
                            return new Test(test_func);
                          });
}

unsigned int GetRandomSeedForTest(const char* env_var) {
  unsigned int seed;
  if (auto env_seed = internal::GetEnv(env_var)) {
    if (absl::SimpleAtoi(*env_seed, &seed)) {
      ABSL_LOG(INFO) << "Using deterministic random seed " << env_var << "="
                     << seed;
      return seed;
    }
  }
  seed = std::random_device()();
  ABSL_LOG(INFO) << "Define environment variable " << env_var << "=" << seed
                 << " for deterministic seeding";
  return seed;
}

}  // namespace internal
}  // namespace tensorstore
