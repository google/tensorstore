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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_INFO_H_
#define TENSORSTORE_INTERNAL_OS_FILE_INFO_H_

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#endif

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/file_descriptor.h"

// Include system headers last to reduce impact of macros.
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#endif

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {

/// Representation of file metadata.
struct FileInfo {
#ifdef _WIN32
  ::BY_HANDLE_FILE_INFORMATION impl = {};
#else
  struct ::stat impl = {};
#endif

  /// Returns `true` when this is a regular file (rather than a
  /// directory or other special file).
  bool IsRegularFile() const;
  /// Returns `true` when this is a directory.
  bool IsDirectory() const;
  /// Returns the size in bytes.
  uint64_t GetSize() const;
  /// Returns a unique identifier of the device/filesystem.
  uint64_t GetDeviceId() const;
  uint64_t GetFileId() const;
  /// Returns the last modified time.
  absl::Time GetMTime() const;
  /// Returns the creation time.
  absl::Time GetCTime() const;
  /// Returns the mode bits.
  uint32_t GetMode() const;
};

/// Retrieves the metadata for an open file.
///
/// \param fd Open file descriptor.
/// \param info[out] Non-null pointer to location where metadata will be stored.
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info);

/// Retrieves the metadata for a file.
absl::Status GetFileInfo(const std::string& path, FileInfo* info);

#ifndef _WIN32
inline bool FileInfo::IsRegularFile() const { return S_ISREG(impl.st_mode); }
inline bool FileInfo::IsDirectory() const { return S_ISDIR(impl.st_mode); }
inline uint64_t FileInfo::GetSize() const { return impl.st_size; }
inline uint64_t FileInfo::GetDeviceId() const { return impl.st_dev; }
inline uint64_t FileInfo::GetFileId() const { return impl.st_ino; }
#else
inline bool FileInfo::IsRegularFile() const {
  return !(impl.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}
inline bool FileInfo::IsDirectory() const {
  return (impl.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}
inline uint64_t FileInfo::GetSize() const {
  return (static_cast<int64_t>(impl.nFileSizeHigh) << 32) +
         static_cast<int64_t>(impl.nFileSizeLow);
}
inline uint64_t FileInfo::GetDeviceId() const {
  return impl.dwVolumeSerialNumber;
}
inline uint64_t FileInfo::GetFileId() const {
  return (static_cast<uint64_t>(impl.nFileIndexHigh) << 32) |
         static_cast<uint64_t>(impl.nFileIndexLow);
}
#endif

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_INFO_H_
