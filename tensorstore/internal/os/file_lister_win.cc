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

#ifndef _WIN32
#error "Use file_lister_posix.cc instead."
#endif

#include "tensorstore/internal/os/file_lister.h"
// Maintain include ordering here:

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/wstring.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {

struct ListerEntry::Impl {
  const std::string& path;
  const std::wstring& wpath;
  std::string_view component;  // NULL-terminated.
  std::wstring_view wcomponent;
  int64_t size;
  bool is_directory;
};

bool ListerEntry::IsDirectory() { return impl_->is_directory; }

const std::string& ListerEntry::GetFullPath() { return impl_->path; };

std::string_view ListerEntry::GetPathComponent() { return impl_->component; }

int64_t ListerEntry::GetSize() {
  return impl_->is_directory ? -1 : impl_->size;
}

absl::Status ListerEntry::Delete() {
  if (impl_->is_directory) {
    if (::RemoveDirectoryW(impl_->wpath.c_str())) {
      return absl::OkStatus();
    }
    return StatusFromOsError(::GetLastError(), "Failed to remove directory: ",
                             QuoteString(impl_->path));
  }
  return DeleteFile(impl_->path);
}

namespace {

struct StackEntry {
  StackEntry(std::string path, std::wstring wpath, size_t component_size,
             size_t wcomponent_size, int64_t size, DWORD attributes)
      : path(std::move(path)),
        wpath(std::move(wpath)),
        component_size(component_size),
        wcomponent_size(wcomponent_size),
        size(size),
        attributes(attributes) {}

  StackEntry& operator=(const StackEntry& other) = delete;
  StackEntry(const StackEntry& other) = delete;
  StackEntry& operator=(StackEntry&& other) {
    path = std::move(other.path);
    wpath = std::move(other.wpath);
    component_size = other.component_size;
    wcomponent_size = other.wcomponent_size;
    size = other.size;
    attributes = other.attributes;
    find_handle = std::exchange(other.find_handle, INVALID_HANDLE_VALUE);
    return *this;
  }
  StackEntry(StackEntry&& other) { *this = std::move(other); }
  ~StackEntry() {
    if (find_handle != INVALID_HANDLE_VALUE) {
      ::FindClose(find_handle);
    }
  }

  // caller variables.
  std::string path;
  std::wstring wpath;
  size_t component_size;
  size_t wcomponent_size;
  int64_t size;
  DWORD attributes;

  // callee variables.
  HANDLE find_handle = INVALID_HANDLE_VALUE;
};

absl::Status RecursiveListImpl(
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item,
    std::string root_directory, std::wstring wroot_directory) {
  ::WIN32_FIND_DATAW find_data;
  size_t path_component_size = 0;
  char path_component_utf8[MAX_PATH * 3];

  std::vector<StackEntry> stack;
  stack.reserve(16);

  // NOTE: Initial entry guaranteed to be a directory by the caller.
  stack.push_back(StackEntry(std::move(root_directory),
                             std::move(wroot_directory), 0, 0, -1,
                             FILE_ATTRIBUTE_DIRECTORY));
  while (!stack.empty()) {
    auto& entry = stack.back();

    // helper to invoke on_item with the correct parameters.
    auto visit = [&]() -> absl::Status {
      std::string_view component(entry.path);
      component.remove_prefix(entry.path.size() - entry.component_size);
      std::wstring_view wcomponent(entry.wpath);
      wcomponent.remove_prefix(entry.wpath.size() - entry.wcomponent_size);
      const bool is_directory = (entry.attributes & FILE_ATTRIBUTE_DIRECTORY);
      ListerEntry::Impl impl{entry.path, entry.wpath, component,
                             wcomponent, entry.size,  is_directory};
      return on_item(ListerEntry(&impl));
    };

    // Initial state: Not a directory.
    if (!(entry.attributes & FILE_ATTRIBUTE_DIRECTORY)) {
      // Not a directory.
      TENSORSTORE_RETURN_IF_ERROR(visit());
      stack.pop_back();
      continue;
    }

    // Initial state: Attempt to open directory.
    if (entry.find_handle == INVALID_HANDLE_VALUE) {
      if (!recurse_into(entry.path)) {
        TENSORSTORE_RETURN_IF_ERROR(visit());
        stack.pop_back();
        continue;
      }

      std::wstring search_path;
      if (entry.wpath.empty()) {
        search_path = L"./*";
      } else {
        search_path = entry.wpath + L"/*";
      }

      entry.find_handle =
          ::FindFirstFileExW(search_path.c_str(), FindExInfoBasic, &find_data,
                             FindExSearchNameMatch,
                             /*lpSearchFilter=*/nullptr,
                             /*dwAdditionalFlags=*/0);

      if (entry.find_handle == INVALID_HANDLE_VALUE) {
        return StatusFromOsError(::GetLastError(), "Failed listing directory: ",
                                 QuoteString(entry.path));
      }
    } else {
      // Try to advance to next entry.
      if (::FindNextFileW(entry.find_handle, &find_data) != TRUE) {
        // Done.
        TENSORSTORE_RETURN_IF_ERROR(visit());
        stack.pop_back();
        continue;
      }
    }

    // It's a valid file. Probably.
    if (wcscmp(find_data.cFileName, L".") == 0 ||
        wcscmp(find_data.cFileName, L"..") == 0) {
      continue;
    }

    // Translate the component name from wchar to utf-8.
    int utf8_size = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, find_data.cFileName, -1,
        path_component_utf8, MAX_PATH * 3, nullptr, nullptr);
    if (utf8_size > 0) {
      path_component_utf8[utf8_size] = 0;
      path_component_size = utf8_size - 1;
    } else if (utf8_size == 0) {
      return StatusFromOsError(::GetLastError(), "Failed listing directory: ",
                               QuoteString(entry.path));
    }
    std::string_view subdir_component(path_component_utf8, path_component_size);
    std::wstring_view subdir_wcomponent(find_data.cFileName);

    // Compose reusable path entry strings.
    std::string subdir_path = entry.path;
    std::wstring subdir_wpath = entry.wpath;
    if (!entry.path.empty() && !absl::EndsWith(entry.path, "/") &&
        !absl::EndsWith(entry.path, "\\")) {
      subdir_path.append("/");
      subdir_wpath.append(L"/");
    }
    subdir_path.append(subdir_component);
    subdir_wpath.append(subdir_wcomponent);

    int64_t size = (static_cast<int64_t>(find_data.nFileSizeHigh) << 32) +
                   static_cast<int64_t>(find_data.nFileSizeLow);

    // "Recursive" call
    stack.push_back(
        StackEntry(subdir_path, subdir_wpath, subdir_component.size(),
                   subdir_wcomponent.size(), size, find_data.dwFileAttributes));
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status RecursiveFileList(
    std::string root_directory,
    absl::FunctionRef<bool(std::string_view)> recurse_into,
    absl::FunctionRef<absl::Status(ListerEntry)> on_item) {
  std::wstring wpath;
  if (root_directory == std::string_view(".")) {
    // Opening CWD.
    wpath = L".";
  } else if (!root_directory.empty()) {
    // Opening a directory other than the CWD.
    TENSORSTORE_RETURN_IF_ERROR(
        internal::ConvertUTF8ToWindowsWide(root_directory, wpath));
    DWORD attributes = ::GetFileAttributesW(wpath.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
      // File does not exist.
      return absl::OkStatus();
    }
    if (!(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
      return absl::NotFoundError(absl::StrCat("Cannot list non-directory: ",
                                              QuoteString(root_directory)));
    }
  }

  auto status = RecursiveListImpl(recurse_into, on_item, root_directory, wpath);
  MaybeAddSourceLocation(status);
  return status;
}

}  // namespace internal_os
}  // namespace tensorstore
