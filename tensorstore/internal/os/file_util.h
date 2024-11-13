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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_UTIL_H_
#define TENSORSTORE_INTERNAL_OS_FILE_UTIL_H_

/// \file Filesystem operations needed by the "file" driver and other
/// tensorstore uses.

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/unique_handle.h"
#include "tensorstore/util/result.h"

// Include system headers last to reduce impact of macros.
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {

#ifdef _WIN32
// Specializations for Windows.

/// Representation of open file/directory.
using FileDescriptor = HANDLE;  // HANDLE

/// File descriptor traits for use with `UniqueHandle`.
struct FileDescriptorTraits {
  static FileDescriptor Invalid() { return ((FileDescriptor)-1); }
  static void Close(FileDescriptor fd);
};

/// Representation of file metadata.
using FileInfo = ::BY_HANDLE_FILE_INFORMATION;

constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }

#else
// Specializations for Posix.

/// Representation of open file/directory.
using FileDescriptor = int;

/// File descriptor traits for use with `UniqueHandle`.
struct FileDescriptorTraits {
  static FileDescriptor Invalid() { return -1; }
  static void Close(FileDescriptor fd);
};

/// Representation of file metadata.
typedef struct ::stat FileInfo;

constexpr inline bool IsDirSeparator(char c) { return c == '/'; }

#endif

/// Suffix used for lock files.
inline constexpr std::string_view kLockSuffix = ".__lock";

/// Unique handle to an open file descriptor.
///
/// The file descriptor is closed automatically by the destructor.
using UniqueFileDescriptor = UniqueHandle<FileDescriptor, FileDescriptorTraits>;

/// --------------------------------------------------------------------------

class MappedRegion;

/// Returns the default page size for MemmapFileReadOnly; offset must be a
/// multiple of this value.
uint32_t GetDefaultPageSize();

/// Returns a Cord containing the contents of the file.
///
/// \param fd File descriptor opened with `OpenFlags::OpenReadOnly`. The
///     FileDescriptor may be closed after calling this function.
/// \param offset Byte offset within file at which to start reading.
/// \param size Number of bytes to read, or 0 to read the entire file.
/// \returns The contents of the file or a failure absl::Status code.
Result<MappedRegion> MemmapFileReadOnly(FileDescriptor fd, size_t offset,
                                        size_t size);

class MappedRegion {
 public:
  // ::munmap happens when the MappedRegion destructor runs.
  ~MappedRegion();

  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;

  MappedRegion(MappedRegion&& other) { *this = std::move(other); }
  MappedRegion& operator=(MappedRegion&& other) {
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    return *this;
  }

  std::string_view as_string_view() const {
    return std::string_view(data_, size_);
  }

  absl::Cord as_cord() && {
    std::string_view string_view = as_string_view();
    data_ = nullptr;
    size_ = 0;
    return absl::MakeCordFromExternal(
        string_view, [](auto s) { MappedRegion cleanup(s.data(), s.size()); });
  }

 private:
  MappedRegion(const char* data, size_t size) : data_(data), size_(size) {}

  friend Result<MappedRegion> MemmapFileReadOnly(FileDescriptor fd, size_t,
                                                 size_t);

  const char* data_;
  size_t size_;
};

/// --------------------------------------------------------------------------

// Restricted subset of POSIX open flags.
enum class OpenFlags : int {
  OpenReadOnly = O_RDONLY,
  OpenWriteOnly = O_WRONLY,
  OpenReadWrite = O_RDWR,
  Create = O_CREAT,
  Append = O_APPEND,
  Exclusive = O_EXCL,

  DefaultWrite = O_CREAT | O_WRONLY,
};

inline OpenFlags operator|(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) | static_cast<int>(b));
}
inline OpenFlags operator&(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) & static_cast<int>(b));
}

/// Wrapper around ::open or the equivalent CreateFileEx call in windows.
///
/// \returns The open file descriptor or a failure absl::Status code.
Result<UniqueFileDescriptor> OpenFileWrapper(const std::string& path,
                                             OpenFlags flags);

inline Result<UniqueFileDescriptor> OpenExistingFileForReading(
    const std::string& path) {
  return OpenFileWrapper(path, OpenFlags::OpenReadOnly);
}

/// Reads from an open file.
///
/// \param fd Open file descriptor.
/// \param buf[out] Pointer to memory where data will be stored.
/// \param count Maximum number of bytes to read.
/// \param offset Byte offset within file at which to start reading.
/// \returns Number of bytes read or a failure absl::Status code.
Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset);

/// Writes to an open file.
///
/// \param fd Open file descriptor.
/// \param buf[in] Pointer to data to write.
/// \param count Maximum number of bytes to write.
/// \returns Number of bytes written or a failure absl::Status code.
Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf, size_t count);

/// Writes an absl::Cord to an open file.
///
/// \param fd Open file descriptor.
/// \param cord[in] data to write.
/// \param count Maximum number of bytes to write.
/// \returns Number of bytes written or a failure absl::Status code.
Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value);

/// Truncates an open file.
///
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status TruncateFile(FileDescriptor fd);

/// Renames an open file.
///
/// \param fd The open file descriptor (ignored by POSIX implementation).
/// \param old_name The existing path.
/// \param new_name The new path.
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name);

/// Deletes an open file.
///
/// \param fd The open file descriptor (ignored by POSIX implementation).
/// \param path The path to the open file.
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path);

/// Deletes a file.
///
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status DeleteFile(const std::string& path);

/// Syncs an open file descriptor.
///
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status FsyncFile(FileDescriptor fd);

/// Acquires a lock on an open file descriptor.
///
/// \returns An unlock function on success, or an error status.
using UnlockFn = void (*)(FileDescriptor fd);
Result<UnlockFn> AcquireFdLock(FileDescriptor fd);

/// --------------------------------------------------------------------------

/// Retrieves the metadata for an open file.
///
/// \param fd Open file descriptor.
/// \param info[out] Non-null pointer to location where metadata will be stored.
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info);

/// Retrieves the metadata for a file.
absl::Status GetFileInfo(const std::string& path, FileInfo* info);

/// Returns `true` if `info` is the metadata of a regular file (rather than a
/// directory or other special file).
inline bool IsRegularFile(const FileInfo& info) {
#ifdef _WIN32
  return !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  return S_ISREG(info.st_mode);
#endif
}

/// Returns `true` if `info` is the metadata of a regular file (rather than a
/// directory or other special file).
inline bool IsDirectory(const FileInfo& info) {
#ifdef _WIN32
  return (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  return S_ISDIR(info.st_mode);
#endif
}

/// Returns the size in bytes.
inline uint64_t GetSize(const FileInfo& info) {
#ifdef _WIN32
  return (static_cast<int64_t>(info.nFileSizeHigh) << 32) +
         static_cast<int64_t>(info.nFileSizeLow);
#else
  return info.st_size;
#endif
}

/// Returns a unique identifier of the device/filesystem.
inline auto GetDeviceId(const FileInfo& info) {
#ifdef _WIN32
  return info.dwVolumeSerialNumber;
#else
  return info.st_dev;
#endif
}

/// Returns a unique identifier of the file (within the filesystem/device).
inline uint64_t GetFileId(const FileInfo& info) {
#ifdef _WIN32
  return (static_cast<uint64_t>(info.nFileIndexHigh) << 32) |
         static_cast<uint64_t>(info.nFileIndexLow);
#else
  return info.st_ino;
#endif
}

/// Returns the last modified time.
inline absl::Time GetMTime(const FileInfo& info) {
#ifdef _WIN32
  // Windows FILETIME is the number of 100-nanosecond intervals since the
  // Windows epoch (1601-01-01) which is 11644473600 seconds before the unix
  // epoch (1970-01-01).
  uint64_t windowsTicks =
      (static_cast<uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      static_cast<uint64_t>(info.ftLastWriteTime.dwLowDateTime);

  return absl::UnixEpoch() +
         absl::Seconds((windowsTicks / 10000000) - 11644473600ULL) +
         absl::Nanoseconds(windowsTicks % 10000000);
#else
#if defined(__APPLE__)
  const struct ::timespec t = info.st_mtimespec;
#else
  const struct ::timespec t = info.st_mtim;
#endif
  return absl::FromTimeT(t.tv_sec) + absl::Nanoseconds(t.tv_nsec);
#endif
}

/// --------------------------------------------------------------------------

/// Opens a directory.
///
/// \returns A valid file descriptor on success, or an invalid file descriptor
///     in the case of an error (in which case `GetLastErrorCode()` retrieves
///     the error).
Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path);

/// Makes a directory.
///
/// \returns `true` on success or if the directory already exists.  Returns
///     `false` in the case of an error (in which case `GetLastErrorCode()`
///     retrieves the error).
absl::Status MakeDirectory(const std::string& path);

/// Syncs an open directory descriptor.
///
/// \returns `true` on success, `false` on error (call `GetLastErrorCode()` to
///     retrieve the error).
absl::Status FsyncDirectory(FileDescriptor fd);

#ifdef _WIN32
/// Returns the Windows temporary directory.
Result<std::string> GetWindowsTempDir();
#endif

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_UTIL_H_
