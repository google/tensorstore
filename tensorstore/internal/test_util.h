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

#ifndef TENSORSTORE_INTERNAL_TEST_UTIL_H_
#define TENSORSTORE_INTERNAL_TEST_UTIL_H_

#include <functional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal {

// Recursively enumerate all the paths starting at the given directory.
absl::Status EnumeratePaths(
    const std::string& directory,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_directory_entry);

// Create a temporary scoped directory.  When the object
// goes out of scope, the directory and all files within are deleted.
class ScopedTemporaryDirectory {
 public:
  ScopedTemporaryDirectory();
  ScopedTemporaryDirectory(const ScopedTemporaryDirectory&) = delete;
  ~ScopedTemporaryDirectory();

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Sets current working directory while in scope.
//
// The destructor restores the existing current working directory.
//
// On Windows, the directory path is specified as UTF-8.
class ScopedCurrentWorkingDirectory {
 public:
  ScopedCurrentWorkingDirectory(const std::string& new_cwd);
  ScopedCurrentWorkingDirectory(const ScopedCurrentWorkingDirectory&) = delete;
  ~ScopedCurrentWorkingDirectory();

 private:
  std::string old_cwd_;
};

/// Registers a GoogleTest case dynamically.
void RegisterGoogleTestCaseDynamically(std::string test_suite_name,
                                       std::string test_name,
                                       std::function<void()> test_func,
                                       SourceLocation loc);

/// Returns a seed for a random number generator, which will either be obtained
/// from `env_var` if specified, or randomly generated and logged.
unsigned int GetRandomSeedForTest(const char* env_var);

}  // namespace internal
}  // namespace tensorstore
#endif  // TENSORSTORE_INTERNAL_TEST_UTIL_H_
