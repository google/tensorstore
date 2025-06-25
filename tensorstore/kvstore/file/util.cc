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

#include "tensorstore/kvstore/file/util.h"

#include <stddef.h>

#include <cassert>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::UniqueFileDescriptor;

namespace tensorstore {
namespace internal_file_util {

/// A key is valid if its consists of one or more '/'-separated non-empty valid
/// path components, where each valid path component does not contain '\0', and
/// is not equal to "." or "..", and does not end in lock_suffix.
bool IsKeyValid(std::string_view key, std::string_view lock_suffix) {
  if (absl::StrContains(key, '\0')) return false;
  if (key.empty()) return false;
  // Do not allow `key` to end with '/'.
  if (key.back() == '/' || key.back() == '\\') {
    return false;
  }
  // Remove leading / which leads to an empty path component.
  if (key.front() == '/' || key.front() == '\\') {
    key = key.substr(1);
  }
  for (std::string_view component :
       absl::StrSplit(key, absl::ByAnyChar("/\\"))) {
    if (component.empty()) return false;
    if (component == ".") return false;
    if (component == "..") return false;
    if (!lock_suffix.empty() && component.size() >= lock_suffix.size() &&
        absl::EndsWith(component, lock_suffix)) {
      return false;
    }
  }
  return true;
}

std::string_view LongestDirectoryPrefix(const KeyRange& range) {
  std::string_view prefix = tensorstore::LongestPrefix(range);
  const size_t i = prefix.rfind('/');
  if (i == std::string_view::npos) return {};
  return prefix.substr(0, i);
}

/// Creates any directory ancestors of `path` that do not exist, and returns an
/// open file descriptor to the parent directory of `path`.
Result<UniqueFileDescriptor> OpenParentDirectory(std::string path) {
  size_t end_pos = path.size();
  Result<UniqueFileDescriptor> fd;
  // Remove path components until we find a directory that exists.
  while (true) {
    // Loop backward until we reach a directory we can open (or .).
    // Loop forward, making directories, until we are done.
    size_t separator_pos = end_pos;
    while (separator_pos != 0 &&
           !internal_os::IsDirSeparator(path[separator_pos - 1])) {
      --separator_pos;
    }
    --separator_pos;
    const char* dir_path;
    if (separator_pos == std::string::npos) {
      dir_path = ".";
    } else if (separator_pos == 0) {
      dir_path = "/";
    } else {
      // Temporarily modify path to make `path.c_str()` a NULL-terminated string
      // containing the current ancestor directory path.
      path[separator_pos] = '\0';
      dir_path = path.c_str();
      end_pos = separator_pos;
    }
    fd = internal_os::OpenDirectoryDescriptor(dir_path);
    if (!fd.ok()) {
      if (absl::IsNotFound(fd.status())) {
        assert(separator_pos != 0 && separator_pos != std::string::npos);
        end_pos = separator_pos - 1;
        continue;
      }
      return fd.status();
    }
    // Revert the change to `path`.
    if (dir_path == path.c_str()) path[separator_pos] = '/';
    break;
  }

  // Add path components and attempt to `mkdir` until we have reached the full
  // path.
  while (true) {
    size_t separator_pos = path.find('\0', end_pos);
    if (separator_pos == std::string::npos) {
      // No more ancestors remain.
      return fd;
    }
    TENSORSTORE_RETURN_IF_ERROR(internal_os::MakeDirectory(path));
    fd = internal_os::OpenDirectoryDescriptor(path);
    TENSORSTORE_RETURN_IF_ERROR(fd.status());
    path[separator_pos] = '/';
    end_pos = separator_pos + 1;
  }
}

Result<absl::Cord> ReadFromFileDescriptor(FileDescriptor fd,
                                          ByteRange byte_range) {
  assert(fd != internal_os::FileDescriptorTraits::Invalid());
  // Large reads could use hugepage-aware memory allocations.
  internal::FlatCordBuilder buffer(byte_range.size(), 0);
  size_t offset = 0;
  while (buffer.available() > 0) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto n, internal_os::PReadFromFile(fd, buffer.available_span(),
                                           byte_range.inclusive_min + offset));
    if (n > 0) {
      offset += n;
      buffer.set_inuse(offset);
      continue;
    }
    if (n == 0) {
      return absl::UnavailableError("Length changed while reading");
    }
  }
  return std::move(buffer).Build();
}

}  // namespace internal_file_util
}  // namespace tensorstore
