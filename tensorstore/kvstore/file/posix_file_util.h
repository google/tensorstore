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

#ifndef TENSORSTORE_KVSTORE_FILE_POSIX_FILE_UTIL_H_
#define TENSORSTORE_KVSTORE_FILE_POSIX_FILE_UTIL_H_

/// \file Implements filesystem operations needed by the "file" driver for
/// Linux/OS X.
///
/// This file also serves as interface documentation for the common interface
/// provided by `posix_file_util.h` and `windows_file_util.h`.

#ifndef _WIN32

#include <string_view>

#include "absl/strings/cord.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/kvstore/file/unique_handle.h"
#include "tensorstore/util/result.h"

// Include system headers last to reduce impact of macros.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef __linux__
#include <sys/file.h>
#endif

namespace tensorstore {
namespace internal_file_util {

/// Suffix used for lock files.
inline constexpr std::string_view kLockSuffix = ".__lock";

/// Representation of open file/directory.
using FileDescriptor = int;

/// File descriptor traits for use with `internal::UniqueHandle`.
struct FileDescriptorTraits {
  static const int Invalid() { return -1; }
  static void Close(int fd) { ::close(fd); }
};

/// Unique handle to an open file descriptor.
///
/// The file descriptor is closed automatically by the destructor.
using UniqueFileDescriptor =
    internal::UniqueHandle<FileDescriptor, FileDescriptorTraits>;

/// Returns `true` if `c` is a directory separator character.
constexpr inline bool IsDirSeparator(char c) { return c == '/'; }

/// Representation of file metadata.
typedef struct ::stat FileInfo;

/// Retrieves the metadata for an open file.
///
/// \param fd Open file descriptor.
/// \param info[out] Non-null pointer to location where metadata will be stored.
/// \returns `true` on success, `false` on error (in which case
///     `GetLastErrorCode()` retrieves the error).
inline bool GetFileInfo(FileDescriptor fd, FileInfo* info) {
  return ::fstat(fd, info) == 0;
}

/// Returns `true` if `info` is the metadata of a regular file (rather than a
/// directory or other special file).
inline bool IsRegularFile(const FileInfo& info) {
  return S_ISREG(info.st_mode);
}

/// Returns the size in bytes.
inline std::uint64_t GetSize(const FileInfo& info) { return info.st_size; }

/// Returns a unique identifier of the device/filesystem.
inline const auto GetDeviceId(const FileInfo& info) { return info.st_dev; }

/// Returns a unique identifier of the file (within the filesystem/device).
inline const auto GetFileId(const FileInfo& info) { return info.st_ino; }

struct PosixMTime {
  std::int64_t sec;
  std::int64_t nsec;
};

/// Returns the last modified time.
inline const PosixMTime GetMTime(const FileInfo& info) {
#ifdef __APPLE__
  const struct ::timespec t = info.st_mtimespec;
#else
  const struct ::timespec t = info.st_mtim;
#endif
  return {t.tv_sec, t.tv_nsec};
}

// File traits.
struct FileLockTraits {
  // technically any value <0 is invalid.
  static const int Invalid() { return -1; }

  // Release the lock on an open file descriptor.
  static void Close(int fd);

  /// Acquires a lock on an open file descriptor.
  ///
  /// On POSIX, the lock is released automatically when the file descriptor is
  /// closed.
  ///
  /// On Windows, the lock is released when the `FileLock` object is destroyed.
  /// It is also released automatically when the file descriptor is closed, but
  /// according to the Windows API documentation, there may be a delay.
  ///
  /// \returns `true` on success, or `false` in case of an error (in which case
  ///     `GetLastErrorCode()` retrieves the error).
  static bool Acquire(int fd);
};

/// Opens an file that must already exist for reading.
///
/// \returns The open file descriptor on success.  An error is indicated by an
///     invalid file descriptor (in which case `GetLastErrorCode()` retrieves
///     the error).
inline UniqueFileDescriptor OpenExistingFileForReading(const char* path) {
  return UniqueFileDescriptor(::open(path, O_RDONLY | O_CLOEXEC));
}

/// Opens a file for writing.  If there is an existing file, it is truncated.
///
/// \returns The open file descriptor on success.  An error is indicated by an
///     invalid file descriptor (in which case `GetLastErrorCode()` retrieves
///     the error).
UniqueFileDescriptor OpenFileForWriting(const std::string& path);

/// Reads from an open file.
///
/// \param fd Open file descriptor.
/// \param buf[out] Pointer to memory where data will be stored.
/// \param count Maximum number of bytes to read.
/// \param offset Byte offset within file at which to start reading.
/// \returns Number of bytes read on success.  Returns `0` to indicate end of
///     file, or `-1` to indicate an error (in which case `GetLastErrorCode()`
///     retrieves the error).
inline std::ptrdiff_t ReadFromFile(FileDescriptor fd, void* buf,
                                   std::size_t count, std::int64_t offset) {
  return ::pread(fd, buf, count, static_cast<off_t>(offset));
}

/// Writes to an open file.
///
/// \param fd Open file descriptor.
/// \param buf[in] Pointer to data to write.
/// \param count Maximum number of bytes to write.
/// \returns Number of bytes written on success.  Returns `0` or `-1` to
///     indicate an error (in which case `GetLastErrorCode()` retrieves the
///     error).
inline std::ptrdiff_t WriteToFile(FileDescriptor fd, const void* buf,
                                  std::size_t count) {
  std::ptrdiff_t n = ::write(fd, buf, count);
  if (count != 0 && n == 0) {
    errno = ENOSPC;
  }
  return n;
}

/// Writes an absl::Cord to an open file.
///
/// \param fd Open file descriptor.
/// \param cord[in] data to write.
/// \param count Maximum number of bytes to write.
/// \returns Number of bytes written on success.  Returns `0` or `-1` to
///     indicate an error (in which case `GetLastErrorCode()` retrieves the
///     error).
std::ptrdiff_t WriteCordToFile(FileDescriptor fd, absl::Cord value);

/// Truncates an open file.
///
/// \returns `true` on success, or `false` in case of an error (in which case
///     `GetLastErrorCode()` retrieves the error).
inline bool TruncateFile(FileDescriptor fd) { return ::ftruncate(fd, 0) == 0; }

/// Renames an open file.
///
/// \param fd The open file descriptor (ignored by POSIX implementation).
/// \param old_name The existing path.
/// \param new_name The new path.
/// \returns `true` on success, or `false` in case of an error (in which case
///     `GetLastErrorCode()` retrieves the error).
inline bool RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                           const std::string& new_name) {
  return ::rename(old_name.c_str(), new_name.c_str()) == 0;
}

/// Deletes an open file.
///
/// \param fd The open file descriptor (ignored by POSIX implementation).
/// \param path The path to the open file.
/// \returns `true` on success, or `false` in case of an error (in which case
///     `GetLastErrorCode()` retrieves the error).
inline bool DeleteOpenFile(FileDescriptor fd, const std::string& path) {
  return ::unlink(path.c_str()) == 0;
}

/// Deletes a file.
///
/// \returns `true` on success, or `false` in case of an error (in which case
///     `GetLastErrorCode()` retrieves the error).
inline bool DeleteFile(const std::string& path) {
  return ::unlink(path.c_str()) == 0;
}

/// Opens a directory.
///
/// \returns A valid file descriptor on success, or an invalid file descriptor
///     in the case of an error (in which case `GetLastErrorCode()` retrieves
///     the error).
inline FileDescriptor OpenDirectoryDescriptor(const char* path) {
  return ::open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
}

/// Makes a directory.
///
/// \returns `true` on success or if the directory already exists.  Returns
///     `false` in the case of an error (in which case `GetLastErrorCode()`
///     retrieves the error).
inline bool MakeDirectory(const char* path) {
  return ::mkdir(path, 0777) == 0 || errno == EEXIST;
}

/// Syncs an open file descriptor.
///
/// \returns `true` on success, `false` on error (call `GetLastErrorCode()` to
///     retrieve the error).
inline bool FsyncFile(FileDescriptor fd) { return ::fsync(fd) == 0; }

/// Syncs an open directory descriptor.
///
/// \returns `true` on success, `false` on error (call `GetLastErrorCode()` to
///     retrieve the error).
inline bool FsyncDirectory(FileDescriptor fd) { return ::fsync(fd) == 0; }

struct DirectoryDeleter {
  void operator()(::DIR* d) { ::closedir(d); }
};
using UniqueDir = std::unique_ptr<::DIR, DirectoryDeleter>;

/// Iterates over a directory.
class DirectoryIterator {
 public:
  /// Represents an entry within a directory that is being iterated.
  struct Entry {
    /// Deletes the file/directory represented by the entry.
    ///
    /// \param is_directory Specifies whether the entry is a directory.
    /// \returns `true` on success, or `false` in case of an error (in which
    /// case
    ///     `GetLastErrorCode()` retrieves the error).
    bool Delete(bool is_directory) const {
      return ::unlinkat(dir_fd, name, is_directory ? AT_REMOVEDIR : 0) == 0;
    }

    /// Creates an entry from a path alone.  The path must outlive the entry.
    static Entry FromPath(const std::string& path) {
      return {AT_FDCWD, path.c_str()};
    }

    int dir_fd;
    const char* name;
  };

  /// Moves to the next (or first, if not called previously) directory entry.
  ///
  /// \returns `true` on success, or `false` if there are no more entries.
  bool Next();

  /// Returns the final path component of the current directory entry.
  ///
  /// \pre Prior call to `Next()` returned `true`.
  std::string_view path_component() const { return e ? e->d_name : ""; }

  /// Returns `true` if the current entry is a directory.
  ///
  /// \pre Prior call to `Next()` returned `true`.
  bool is_directory() const;

  /// Returns the current directory entry.
  ///
  /// \pre Prior call to `Next()` returned `true`.
  Entry GetEntry() const;

  /// Makes a directory iterator for the path specified by `entry`.
  ///
  /// \param entry The path to the directory.
  /// \param new_iterator[out] Pointer to directory iterator.  If the specified
  ///     directory can be opened successfully, set to a new iterator.
  /// \returns `true` if the directory can be opened successfully, or it does
  ///     not exist; or `false` in the case of another error (in which case
  ///     `GetLastErrorCode()` retrieves the error).
  static bool Make(Entry entry,
                   std::unique_ptr<DirectoryIterator>* new_iterator);

 private:
  UniqueDir dir;
  struct dirent* e;
};

}  // namespace internal_file_util
}  // namespace tensorstore

#endif  // !defined(_WIN32)
#endif  // TENSORSTORE_KVSTORE_FILE_POSIX_FILE_UTIL_H_
