// Copyright 2024 The TensorStore Authors
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

#ifdef _WIN32
#error "Use file_lister_win.cc instead."
#endif

#include "tensorstore/internal/os/file_lister.h"
// Maintain include ordering here:

#include <dirent.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/file_util.h"

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {

struct ListerEntry::Impl {
  int parent_fd;
  const std::string& full_path;
  std::string_view component;  // NULL-terminated.
  bool is_directory;
};

bool ListerEntry::IsDirectory() { return impl_->is_directory; }

const std::string& ListerEntry::GetFullPath() { return impl_->full_path; };

std::string_view ListerEntry::GetPathComponent() { return impl_->component; }

int64_t ListerEntry::GetSize() { return -1; }

absl::Status ListerEntry::Delete() {
  PotentiallyBlockingRegion region;
  if (::unlinkat(impl_->parent_fd, impl_->component.data(),
                 impl_->is_directory ? AT_REMOVEDIR : 0) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno, "Failed to remove ",
                           impl_->is_directory ? "directory: " : "file: ",
                           QuoteString(GetFullPath()));
}

namespace {

struct StackEntry {
  StackEntry(std::string path, size_t component_size, int parent_fd)
      : path(std::move(path)),
        component_size(component_size),
        parent_fd(parent_fd) {}

  StackEntry& operator=(const StackEntry& other) = delete;
  StackEntry(const StackEntry& other) = delete;
  StackEntry& operator=(StackEntry&& other) {
    path = std::move(other.path);
    component_size = other.component_size;
    parent_fd = other.parent_fd;
    fd = std::exchange(other.fd, -1);
    dir = std::exchange(other.dir, nullptr);
    return *this;
  }
  StackEntry(StackEntry&& other) { *this = std::move(other); }
  ~StackEntry() {
    if (dir != nullptr) {
      // closedir closes fd.
      ::closedir(dir);
    } else if (fd != -1) {
      ::close(fd);
    }
  }

  // caller variables
  std::string path;
  size_t component_size;
  int parent_fd;

  // callee variables.
  int fd = -1;
  DIR* dir = nullptr;
};

absl::Status RecursiveListImpl(
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item,
    std::string root_directory) {
  std::vector<StackEntry> stack;
  stack.reserve(16);

  // NOTE: Initial entry guaranteed to be a directory by the caller.
  stack.push_back(StackEntry{std::move(root_directory), 0, AT_FDCWD});
  while (!stack.empty()) {
    auto& entry = stack.back();
    std::string_view component(entry.path);
    component.remove_prefix(entry.path.size() - entry.component_size);

    // helper to invoke on_item with the correct parameters.
    auto visit = [&](bool is_directory) -> absl::Status {
      ListerEntry::Impl impl{entry.parent_fd, entry.path, component,
                             is_directory};
      return on_item(ListerEntry(&impl));
    };

    // Initial state; attempt to open as a directory.
    if (entry.fd == -1) {
      do {
        PotentiallyBlockingRegion region;
        entry.fd = ::openat(entry.parent_fd,
                            component.empty()
                                ? entry.path.empty() ? "." : entry.path.c_str()
                                : component.data(),
                            O_CLOEXEC | O_RDONLY | O_DIRECTORY |
                                (entry.parent_fd == AT_FDCWD ? 0 : O_NOFOLLOW));
      } while (entry.fd == -1 && (errno == EINTR || errno == EAGAIN));

      // Failed to open the directory:
      if (entry.fd == -1) {
        if (errno == ENOTDIR) {
          // Visit file.
          TENSORSTORE_RETURN_IF_ERROR(visit(false));
          stack.pop_back();
          continue;
        }
        if (errno == ENOENT) {
          // Does not exist; ignore.
          stack.pop_back();
          continue;
        }
        return StatusFromOsError(
            errno, "Failed while listing: ", QuoteString(entry.path));
      }
    }

    // Successfully opened directory; attempt to enumerate directory.
    if (entry.dir == nullptr) {
      if (!recurse_into(entry.path)) {
        TENSORSTORE_RETURN_IF_ERROR(visit(true));
        stack.pop_back();
        continue;
      }

      entry.dir = ::fdopendir(entry.fd);
      if (entry.dir == nullptr) {
        return StatusFromOsError(
            errno, "Failed while listing: ", QuoteString(entry.path));
      }
    }

    // Read the next entry.
    struct dirent* e = ::readdir(entry.dir);
    while (e != nullptr &&
           (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)) {
      e = ::readdir(entry.dir);
    }

    if (e == nullptr) {
      // No more entries.
      TENSORSTORE_RETURN_IF_ERROR(visit(true));
      stack.pop_back();
      continue;
    }

    std::string_view subdir_component(e->d_name);
    std::string subdir_path = absl::StrCat(
        entry.path,
        (entry.path.empty() || absl::EndsWith(entry.path, "/")) ? "" : "/",
        subdir_component);

    // "Recursive" call.
    stack.push_back(
        StackEntry{std::move(subdir_path), subdir_component.size(), entry.fd});
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RecursiveFileList(
    std::string root_directory,
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item) {
  // root_directory must be a directory.
  struct ::stat dir_stat;
  if (::fstatat(AT_FDCWD, root_directory.empty() ? "." : root_directory.c_str(),
                &dir_stat, 0) != 0) {
    if (errno == ENOENT) return absl::OkStatus();
    return StatusFromOsError(errno,
                             "Failed to stat: ", QuoteString(root_directory));
  }
  if (!S_ISDIR(dir_stat.st_mode)) {
    return absl::NotFoundError(absl::StrCat("Cannot list non-directory: ",
                                            QuoteString(root_directory)));
  }

  auto status = RecursiveListImpl(recurse_into, on_item, root_directory);
  MaybeAddSourceLocation(status);
  return status;
}

}  // namespace internal_os
}  // namespace tensorstore
