// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/os/filesystem.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // IWYU pragma: keep

#include "absl/log/absl_check.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/path.h"

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
namespace internal_os {

#if TENSORSTORE_USE_STD_FILESYSTEM

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath() {
  std::error_code ec;
  auto base_dir = std::filesystem::temp_directory_path(ec);
  ABSL_CHECK(!ec);
  return base_dir.generic_string();
}

// similar to std::filesystem::create_directory
absl::Status MakeDirectory(const std::string& dirname) {
  std::error_code ec;
  std::filesystem::create_directory(dirname, ec);
  return (!ec) ? absl::OkStatus()
               : internal::StatusFromOsError(ec.value(), " while creating ",
                                             dirname);
}

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& dirname) {
  std::error_code ec;
  std::filesystem::remove_all(dirname, ec);
  return !ec ? absl::OkStatus()
             : internal::StatusFromOsError(ec.value(), " while removing ",
                                           dirname);
}

#else  // !TENSORSTORE_USE_STD_FILESYSTEM

namespace {
absl::Status RemovePathImpl(const std::string& path, bool is_dir) {
  if (::remove(path.c_str()) != 0) {
    return internal::StatusFromOsError(
        errno, is_dir ? " while deleting directory" : " while deleting file");
  }
  return absl::OkStatus();
}

absl::Status EnumeratePathsImpl(
    const std::string& dirname,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_entry) {
  DIR* dir = ::opendir(dirname.c_str());
  if (dir == NULL) {
    return internal::StatusFromOsError(errno, " while opening ", dirname);
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
    std::string_view entry_dname(entry->d_name);
    if (entry_dname == "." || entry_dname == "..") {
      continue;
    }
    std::string path = tensorstore::internal::JoinPath(dirname, entry_dname);
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

}  // namespace

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath() {
  for (char const* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    auto env = internal::GetEnv(variable);
    if (env) return *env;
  }
  return "/tmp";
}

// similar to std::filesystem::create_directory
absl::Status MakeDirectory(const std::string& dirname) {
  auto ret = ::mkdir(dirname.c_str(), 0700);
  return (ret == 0)
             ? absl::OkStatus()
             : internal::StatusFromOsError(ret, " while creating ", dirname);
}

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& dirname) {
  return EnumeratePathsImpl(dirname, [](const std::string& path, bool is_dir) {
    return RemovePathImpl(path, is_dir);
  });
}

#endif  // TENSORSTORE_USE_STD_FILESYSTEM

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

}  // namespace internal_os
}  // namespace tensorstore
