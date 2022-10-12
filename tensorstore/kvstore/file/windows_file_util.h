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

#ifndef TENSORSTORE_KVSTORE_FILE_WINDOWS_FILE_UTIL_H_
#define TENSORSTORE_KVSTORE_FILE_WINDOWS_FILE_UTIL_H_

/// \file
/// Implements filesystem operations needed by the "file" driver for MS Windows.
///
/// Refer to `posix_file_util.h` for function documentation.

#ifdef _WIN32

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/kvstore/file/unique_handle.h"
#include "tensorstore/util/result.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace tensorstore {
namespace internal_file_util {

inline constexpr std::string_view kLockSuffix = ".__lock";

using FileDescriptor = HANDLE;
struct FileDescriptorTraits {
  static const HANDLE Invalid() { return INVALID_HANDLE_VALUE; }
  static void Close(HANDLE handle) { ::CloseHandle(handle); }
};

using UniqueFileDescriptor =
    internal::UniqueHandle<FileDescriptor, FileDescriptorTraits>;

constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }

using FileInfo = ::BY_HANDLE_FILE_INFORMATION;
inline bool GetFileInfo(FileDescriptor fd, FileInfo* info) {
  return static_cast<bool>(::GetFileInformationByHandle(fd, info));
}

inline bool IsRegularFile(const FileInfo& info) {
  return !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}

inline std::uint64_t GetSize(const FileInfo& info) {
  return (static_cast<std::int64_t>(info.nFileSizeHigh) << 32) +
         static_cast<std::int64_t>(info.nFileSizeLow);
}

inline DWORD GetDeviceId(const FileInfo& info) {
  return info.dwVolumeSerialNumber;
}

inline std::uint64_t GetFileId(const FileInfo& info) {
  return (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32) |
         static_cast<std::uint64_t>(info.nFileIndexLow);
}

inline std::uint64_t GetMTime(const FileInfo& info) {
  const ::FILETIME t = info.ftLastWriteTime;
  return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) |
         static_cast<std::uint64_t>(t.dwLowDateTime);
}

struct FileLockTraits {
  static const HANDLE Invalid() { return INVALID_HANDLE_VALUE; }
  static void Close(HANDLE handle);
  static bool Acquire(HANDLE handle);
};

UniqueFileDescriptor OpenExistingFileForReading(std::string_view path);
UniqueFileDescriptor OpenFileForWriting(std::string_view path);

std::ptrdiff_t ReadFromFile(FileDescriptor fd, void* buf, std::size_t count,
                            std::int64_t offset);
std::ptrdiff_t WriteToFile(FileDescriptor fd, const void* buf,
                           std::size_t count);

std::ptrdiff_t WriteCordToFile(FileDescriptor fd, absl::Cord value);

inline bool TruncateFile(FileDescriptor fd) {
  return static_cast<bool>(::SetEndOfFile(fd));
}

bool RenameOpenFile(FileDescriptor fd, std::string_view old_name,
                    std::string_view new_name);

bool DeleteOpenFile(FileDescriptor fd, std::string_view path);
bool DeleteFile(std::string_view path);

FileDescriptor OpenDirectoryDescriptor(const char* path);
bool MakeDirectory(const char* path);

bool FsyncFile(FileDescriptor fd);

// Windows does not support fsync on directories.
inline bool FsyncDirectory(FileDescriptor fd) { return true; }

struct FindHandleTraits {
  static const HANDLE Invalid() { return INVALID_HANDLE_VALUE; }
  static void Close(HANDLE handle) { ::FindClose(handle); }
};

struct DirectoryIterator {
 public:
  struct Entry {
    static Entry FromPath(const std::string& path) { return {path}; }
    bool Delete(bool is_directory) const;

    std::string path;
  };

  void Update(const ::WIN32_FIND_DATAW& find_data);
  bool Next();
  std::string_view path_component() const {
    return std::string_view(path_component_utf8, path_component_size);
  }
  bool is_directory() const { return is_directory_; }

  Entry GetEntry() const;

  static bool Make(Entry entry,
                   std::unique_ptr<DirectoryIterator>* new_iterator);

 private:
  internal::UniqueHandle<HANDLE, internal_file_util::FindHandleTraits>
      find_handle;
  std::string directory_path_;
  bool initial = true;
  DWORD last_error;
  bool is_directory_;
  size_t path_component_size;
  char path_component_utf8[MAX_PATH * 3];
};

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // defined(_WIN32)
#endif  // TENSORSTORE_KVSTORE_FILE_WINDOWS_FILE_UTIL_H_
