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

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/absl_check.h"  // IWYU pragma: keep
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/os/file_lister.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {

// similar to std::filesystem::temp_directory_path
std::string TemporaryDirectoryPath() {
  // Preferentially return an existing directory.
  for (char const* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    auto env = internal::GetEnv(variable);
    internal_os::FileInfo info;
    if (env && internal_os::GetFileInfo(*env, &info).ok() &&
        IsDirectory(info)) {
      return *env;
    }
  }
#ifdef _WIN32
  {
    auto tmpdir = GetWindowsTempDir();
    if (tmpdir.ok()) return *tmpdir;
  }
#endif
  // The directory does not exist; consider creating it?
  for (char const* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    auto env = internal::GetEnv(variable);
    if (env) return *env;
  }
  return "/tmp";
}

// similar to std::filesystem::remove_all
absl::Status RemoveAll(const std::string& root_directory) {
  absl::Status result;
  auto status = RecursiveFileList(
      root_directory, /*recurse_into=*/[](std::string_view) { return true; },
      /*on_item=*/
      [&](auto entry) {
        auto status = entry.Delete();
        if (!status.ok() && !absl::IsNotFound(status)) {
          ABSL_LOG(INFO) << "Failed to remove " << entry.GetFullPath() << ": "
                         << status;
          MaybeAddSourceLocation(status);
          result.Update(status);
        }
        return absl::OkStatus();
      });
  result.Update(status);
  return result;
}

absl::Status EnumeratePaths(
    const std::string& root_directory,
    std::function<absl::Status(const std::string& /*name*/, bool /*is_dir*/)>
        on_directory_entry) {
  absl::Status result;
  auto status = RecursiveFileList(
      root_directory, /*recurse_into=*/[](std::string_view) { return true; },
      /*on_item=*/
      [&](auto entry) {
        auto status =
            on_directory_entry(entry.GetFullPath(), entry.IsDirectory());
        MaybeAddSourceLocation(status);
        result.Update(status);
        return absl::OkStatus();
      });
  result.Update(status);
  return result;
}

/// Returns the list of relative paths contained within the directory `root`.
std::vector<std::string> GetDirectoryContents(
    const std::string& root_directory) {
  std::vector<std::string> paths;
  TENSORSTORE_CHECK_OK(RecursiveFileList(
      root_directory, /*recurse_into=*/[](std::string_view) { return true; },
      /*on_item=*/
      [&](auto entry) {
        const auto& path = entry.GetFullPath();
        if (path != root_directory) {
          paths.emplace_back(path.substr(root_directory.size() + 1));
        }
        return absl::OkStatus();
      }));

  return paths;
}

}  // namespace internal_os
}  // namespace tensorstore
