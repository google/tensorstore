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

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>

#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/status.h"

#if defined(_WIN32)
#define TENSORSTORE_USE_STD_FILESYSTEM 1
#elif !defined(TENSORSTORE_USE_STD_FILESYSTEM)
#define TENSORSTORE_USE_STD_FILESYSTEM 0
#endif

#if TENSORSTORE_USE_STD_FILESYSTEM
// use the new C++ apis
#include <filesystem>
#else

// Include these system headers last to reduce impact of macros.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef __linux__
#include <sys/file.h>
#endif

#endif

namespace tensorstore {
namespace internal {
namespace {

#if TENSORSTORE_USE_STD_FILESYSTEM

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath() {
  std::error_code ec;
  auto base_dir = std::filesystem::temp_directory_path(ec);
  TENSORSTORE_CHECK(!ec);
  return base_dir.generic_string();
}

// similar to std::filesystem::create_directory
bool MakeDirectory(const std::string& dirname) {
  std::error_code ec;
  std::filesystem::create_directory(dirname, ec);
  return !ec;
}

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& dirname) {
  std::error_code ec;
  std::filesystem::remove_all(dirname, ec);
  return !ec ? absl::OkStatus()
             : StatusFromOsError(ec.value(), " while removing ", dirname);
}

#else  // !TENSORSTORE_USE_STD_FILESYSTEM

absl::Status EnumeratePathsImpl(
    const std::string& dirname,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_entry) {
  DIR* dir = ::opendir(dirname.c_str());
  if (dir == NULL) {
    return StatusFromOsError(errno, " while opening ", dirname);
  }

  absl::Status result;
  struct dirent* entry;

  auto is_directory = [&]() {
    if (entry->d_type == DT_UNKNOWN) {
      // In the case of an unknown type, fstat the directory.
      struct ::stat statbuf;
      if (::fstatat(::dirfd(dir), entry->d_name, &statbuf,
                    AT_SYMLINK_NOFOLLOW)) {
        return S_ISDIR(statbuf.st_mode);
      }
      return false;
    }
    return (entry->d_type == DT_DIR);
  };

  while ((entry = ::readdir(dir)) != NULL) {
    absl::string_view entry_dname(entry->d_name);
    if (entry_dname == "." || entry_dname == "..") {
      continue;
    }
    std::string path = StrCat(dirname, "/", entry_dname);
    if (is_directory()) {
      result.Update(EnumeratePathsImpl(path, on_entry));
    } else {
      result.Update(on_entry(path, false));
    }
  }
  ::closedir(dir);
  result.Update(on_entry(dirname, true));
  return result;
}

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath() {
  for (char const* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    auto env = GetEnv(variable);
    if (env) return *env;
  }
  return "/tmp";
}

// similar to std::filesystem::create_directory
bool MakeDirectory(const std::string& dirname) {
  return ::mkdir(dirname.c_str(), 0700) == 0;
}

absl::Status RemovePathImpl(const std::string& path, bool is_dir) {
  if (::remove(path.c_str()) != 0) {
    return StatusFromOsError(
        errno, is_dir ? " while deleting directory" : " while deleting file");
  }
  return absl::OkStatus();
}

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& dirname) {
  return EnumeratePathsImpl(dirname, [](const std::string& path, bool is_dir) {
    return RemovePathImpl(path, is_dir);
  });
}

#endif  // TENSORSTORE_USE_STD_FILESYSTEM

}  // namespace

absl::Status EnumeratePaths(
    const std::string& directory,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_directory_entry) {
#if TENSORSTORE_USE_STD_FILESYSTEM
  absl::Status result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(directory)) {
    result.Update(on_directory_entry(entry.path().generic_string(),
                                     entry.is_directory()));
  }
  return result;
#else
  return EnumeratePathsImpl(directory, on_directory_entry);
#endif
}

ScopedTemporaryDirectory::ScopedTemporaryDirectory() {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";

  absl::BitGen gen;
  char data[24];
  for (auto& x : data) {
    x = kAlphabet[absl::Uniform(gen, 0u, std::size(kAlphabet) - 1)];
  }

  path_ = StrCat(TemporaryDirectoryPath(), "/tmp_tensorstore_test",
                 absl::string_view(data, std::size(data)));

  TENSORSTORE_CHECK(MakeDirectory(path_));
}

ScopedTemporaryDirectory::~ScopedTemporaryDirectory() {
  TENSORSTORE_CHECK_OK(RemoveAll(path_));
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
      TENSORSTORE_LOG("Using deterministic random seed ", env_var, "=", seed);
      return seed;
    }
  }
  seed = std::random_device()();
  TENSORSTORE_LOG("Define environment variable ", env_var, "=", seed,
                  " for deterministic seeding");
  return seed;
}

}  // namespace internal
}  // namespace tensorstore
