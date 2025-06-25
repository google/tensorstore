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

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

// Include system headers last to reduce impact of macros.
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {

// Specializations for Windows.
constexpr inline bool IsDirSeparator(char c) {
#ifdef _WIN32
  return c == '\\' || c == '/';
#else
  return c == '/';
#endif
}

/// Suffix used for lock files.
inline constexpr std::string_view kLockSuffix = ".__lock";

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
Result<MemoryRegion> MemmapFileReadOnly(FileDescriptor fd, size_t offset,
                                        size_t size);

/// --------------------------------------------------------------------------

// Restricted subset of POSIX open flags.
enum class OpenFlags : int {
  OpenReadOnly = O_RDONLY,
  OpenWriteOnly = O_WRONLY,
  OpenReadWrite = O_RDWR,
  Create = O_CREAT,
  Append = O_APPEND,
  Exclusive = O_EXCL,
  CloseOnExec = O_CLOEXEC,
  ReadWriteMask = O_RDONLY | O_WRONLY | O_RDWR,

  DefaultRead = O_RDONLY | O_CLOEXEC,
  DefaultWrite = O_CREAT | O_WRONLY | O_CLOEXEC,
};

inline constexpr OpenFlags operator|(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) | static_cast<int>(b));
}
inline constexpr OpenFlags operator&(OpenFlags a, OpenFlags b) {
  return static_cast<OpenFlags>(static_cast<int>(a) & static_cast<int>(b));
}

/// Wrapper around ::open or the equivalent CreateFileEx call in windows.
///
/// \returns The open file descriptor or a failure absl::Status code.
Result<UniqueFileDescriptor> OpenFileWrapper(const std::string& path,
                                             OpenFlags flags);

/// Attempt to set the flags on an open file descriptor.
///
/// \returns `absl::OkStatus` on success, or a failure absl::Status code.
absl::Status SetFileFlags(FileDescriptor fd, OpenFlags flags);

/// Reads from an open file.
///
/// \param fd Open file descriptor.
/// \param buffer[out] Pointer to memory where data will be stored.
/// \returns Number of bytes read or a failure absl::Status code.
Result<ptrdiff_t> ReadFromFile(FileDescriptor fd,
                               tensorstore::span<char> buffer);

/// Reads from an open file.
///
/// \param fd Open file descriptor.
/// \param buffer[out] Pointer to memory where data will be stored.
/// \param offset Byte offset within file at which to start reading.
/// \returns Number of bytes read or a failure absl::Status code.
Result<ptrdiff_t> PReadFromFile(FileDescriptor fd,
                                tensorstore::span<char> buffer, int64_t offset);

/// Reads the entire file into a string.
///
/// \param fd Open file descriptor.
Result<std::string> ReadAllToString(const std::string& path);

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

/// Waits for a pipe to be readable.
///
/// \param fd Open file descriptor.
/// \param deadline When not absl::InfiniteFuture(), if supported, waits util
///     deadline for a read.
absl::Status AwaitReadablePipe(FileDescriptor fd, absl::Time deadline);

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
