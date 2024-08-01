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

#include "absl/base/optimization.h"
#ifdef _WIN32
#error "Use file_util_win.cc instead."
#endif

#include "tensorstore/internal/os/file_util.h"
// Maintain include ordering here:

#include <dirent.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

// Most modern unix allow 1024 iovs.
#if defined(UIO_MAXIOV)
#define TENSORSTORE_MAXIOV UIO_MAXIOV
#else
#define TENSORSTORE_MAXIOV 1024
#endif

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusFromOsError;

namespace tensorstore {
namespace internal_os {

// On FreeBSD and Mac OS X, `flock` can safely be used instead of open file
// descriptor locks.  `flock`/`fcntl`/`lockf` all use the same underlying lock
// mechanism and are all compatible with each other, and with NFS.
//
// On Linux, `lockf` is simply equivalent to traditional `fcntl` UNIX record
// locking (which is compatible with open file descriptor locks), while `flock`
// is a completely independent mechanism, with some bad NFS interactions: on
// Linux <=2.6.11, `flock` on an NFS-mounted filesystem provides only local
// locking; on Linux >=2.6.12, `flock` on an NFS-mounted filesystem is treated
// as an `fnctl` UNIX record lock that does affect all NFS clients.

#ifdef __linux__
// When building with glibc <2.20, or another libc which predates
// OFD locks, define the constant ourselves.  This assumes that the libc
// and kernel definitions for struct flock are identical.
#ifndef F_OFD_SETLK
#define F_OFD_SETLK 37
#endif
#ifndef F_OFD_SETLKW
#define F_OFD_SETLKW 38
#endif
#endif  // __linux__

namespace {

#if defined(F_OFD_SETLKW)
void UnlockFcntlLock(FileDescriptor fd) {
  // This releases a lock acquired by fcntl(F_OFD_SETLKW).
  // This is not strictly necessary as the posix/linux locks will be released
  // when the fd is closed, but it allows easier reasoning by making locking
  // behave similarly across platforms.
  while (true) {
    struct ::flock lock;
    lock.l_type = F_UNLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_pid = 0;
    {
      PotentiallyBlockingRegion region;
      if (::fcntl(fd, F_OFD_SETLK, &lock) != -1) return;
    }
    if (errno == EINTR) continue;
    ABSL_LOG_FIRST_N(INFO, 1)
        << StatusFromOsError(errno, "Failed to release lock");
    return;
  }
  ABSL_UNREACHABLE();
}
#endif

void UnlockFlockLock(FileDescriptor fd) {
  while (true) {
    {
      PotentiallyBlockingRegion region;
      if (::flock(fd, LOCK_UN) != -1) return;
    }
    if (errno == EINTR) continue;
    ABSL_LOG_FIRST_N(INFO, 1)
        << StatusFromOsError(errno, "Failed to release lock");
    return;
  }
  ABSL_UNREACHABLE();
}

}  // namespace

Result<UnlockFn> AcquireFdLock(FileDescriptor fd) {
#if defined(F_OFD_SETLKW)
  while (true) {
    // This blocks until the lock is acquired (SETLKW).  If any signal is
    // received by the current thread, `fcntl` returns `EINTR`.
    //
    // Note: To support cancellation while waiting on the lock, modify to
    // pass the Promise to this function and register an ExecuteWhenNotNeeded
    // function that uses `pthread_kill` to send a signal to this thread to
    // abort the `fcntl` call (taking care to do it in a race-free way). Use
    // Linux 3.15+ open file descriptor lock.
    struct ::flock lock;
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_pid = 0;
    {
      PotentiallyBlockingRegion region;
      if (::fcntl(fd, F_OFD_SETLKW, &lock) != -1) return UnlockFcntlLock;
    }
    if (errno == EINTR) continue;
    if (errno == EINVAL || errno == ENOTSUP) break;
    return StatusFromOsError(errno, "Failed to lock file");
  }
#endif
  while (true) {
    {
      PotentiallyBlockingRegion region;
      if (::flock(fd, LOCK_EX) != -1) return UnlockFlockLock;
    }
    if (errno == EINTR) continue;
    return StatusFromOsError(errno, "Failed to lock file");
  }
  ABSL_UNREACHABLE();
}

Result<UniqueFileDescriptor> OpenExistingFileForReading(
    const std::string& path) {
  FileDescriptor fd;
  {
    PotentiallyBlockingRegion region;
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  }
  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(errno, "Failed to open: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

Result<UniqueFileDescriptor> OpenFileForWriting(const std::string& path) {
  FileDescriptor fd = FileDescriptorTraits::Invalid();
  const auto attempt_open = [&] {
    PotentiallyBlockingRegion region;
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0666);
  };
#ifndef __APPLE__
  attempt_open();
#else
  // Maximum number of attempts to open the file.  On macOS, `open` may return
  // `ENOENT` or `EPERM` if an existing file is deleted concurrently with the
  // call to `open`.  Mitigate this race condition by retrying, limiting the
  // number of retries to avoid getting stuck in an infinite loop if the error
  // is due to a parent directory having been deleted.
  constexpr int kMaxAttempts = 100;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    attempt_open();
    if (fd != FileDescriptorTraits::Invalid() ||
        (errno != ENOENT && errno != EPERM))
      break;
  }
#endif
  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(errno, "Failed to create: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset) {
  ssize_t n;
  do {
    PotentiallyBlockingRegion region;
    n = ::pread(fd, buf, count, static_cast<off_t>(offset));
  } while ((n < 0) && (errno == EINTR || errno == EAGAIN));
  if (n >= 0) {
    return n;
  }
  return StatusFromOsError(errno, "Failed to read from file");
}

Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf,
                              size_t count) {
  ssize_t n;
  do {
    PotentiallyBlockingRegion region;
    n = ::write(fd, buf, count);
  } while ((n < 0) && (errno == EINTR || errno == EAGAIN));
  if (count != 0 && n == 0) {
    errno = ENOSPC;
  } else if (n >= 0) {
    return n;
  }
  return StatusFromOsError(errno, "Failed to write to file");
}

Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  absl::InlinedVector<iovec, 16> iovs;

  for (std::string_view chunk : value.Chunks()) {
    struct iovec iov;
    iov.iov_base = const_cast<char*>(chunk.data());
    iov.iov_len = chunk.size();
    iovs.emplace_back(iov);
    if (iovs.size() >= TENSORSTORE_MAXIOV) break;
  }
  ssize_t n;
  do {
    PotentiallyBlockingRegion region;
    n = ::writev(fd, iovs.data(), iovs.size());
  } while ((n < 0) && (errno == EINTR || errno == EAGAIN));
  if (!value.empty() && n == 0) {
    errno = ENOSPC;
  } else if (n >= 0) {
    return n;
  }
  return StatusFromOsError(errno, "Failed to write to file");
}

absl::Status TruncateFile(FileDescriptor fd) {
  PotentiallyBlockingRegion region;
  if (::ftruncate(fd, 0) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno, "Failed to truncate file");
}

absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name) {
  PotentiallyBlockingRegion region;
  if (::rename(old_name.c_str(), new_name.c_str()) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno, "Failed to rename: ", QuoteString(old_name),
                           " to: ", QuoteString(new_name));
}

absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path) {
  PotentiallyBlockingRegion region;
  if (::unlink(path.c_str()) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno, "Failed to delete: ", QuoteString(path));
}

absl::Status DeleteFile(const std::string& path) {
  PotentiallyBlockingRegion region;
  if (::unlink(path.c_str()) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno, "Failed to delete: ", QuoteString(path));
}

absl::Status FsyncFile(FileDescriptor fd) {
  PotentiallyBlockingRegion region;
  if (::fsync(fd) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno);
}

absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info) {
  PotentiallyBlockingRegion region;
  if (::fstat(fd, info) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno);
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  PotentiallyBlockingRegion region;
  if (::stat(path.c_str(), info) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno);
}

Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path) {
  FileDescriptor fd;
  {
    PotentiallyBlockingRegion region;
    fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
  }
  if (fd == FileDescriptorTraits::Invalid()) {
    return StatusFromOsError(errno,
                             "Failed to open directory: ", QuoteString(path));
  }
  return UniqueFileDescriptor(fd);
}

absl::Status MakeDirectory(const std::string& path) {
  PotentiallyBlockingRegion region;
  if (::mkdir(path.c_str(), 0777) == 0 || errno == EEXIST) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno,
                           "Failed to create directory: ", QuoteString(path));
}

absl::Status FsyncDirectory(FileDescriptor fd) {
  PotentiallyBlockingRegion region;
  if (::fsync(fd) == 0) {
    return absl::OkStatus();
  }
  return StatusFromOsError(errno);
}

}  // namespace internal_os
}  // namespace tensorstore
