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

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

#if ABSL_HAVE_MMAP
#include <sys/mman.h>
#endif

// Most modern unix allow 1024 iovs.
#if defined(UIO_MAXIOV)
#define TENSORSTORE_MAXIOV UIO_MAXIOV
#else
#define TENSORSTORE_MAXIOV 1024
#endif

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusFromOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

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

namespace tensorstore {
namespace internal_os {
namespace {

auto& mmap_count = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/file/mmap_count",
    internal_metrics::MetricMetadata("Count of total mmap files"));

auto& mmap_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/file/mmap_bytes",
    internal_metrics::MetricMetadata("Count of total mmap bytes"));

auto& mmap_active = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/file/mmap_active",
    internal_metrics::MetricMetadata("Count of active mmap files"));

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");

#if defined(F_OFD_SETLKW)
void UnlockFcntlLock(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});
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
      if (::fcntl(fd, F_OFD_SETLK, &lock) != -1) {
        return;
      }
    }
    if (errno == EINTR) continue;
    tspan.Log("errno", errno);
    return;
  }
  ABSL_UNREACHABLE();
}
#endif

void UnlockFlockLock(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  while (true) {
    {
      PotentiallyBlockingRegion region;
      if (::flock(fd, LOCK_UN) != -1) {
        return;
      }
    }
    if (errno == EINTR) continue;
    tspan.Log("errno", errno);
    return;
  }
  ABSL_UNREACHABLE();
}

}  // namespace

void FileDescriptorTraits::Close(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  while (true) {
    if (::close(fd) == 0) {
      return;
    }
    if (errno == EINTR) continue;
    tspan.Log("errno", errno);
    return;
  }
}

Result<UnlockFn> AcquireFdLock(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

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
      if (::fcntl(fd, F_OFD_SETLKW, &lock) != -1) {
        return UnlockFcntlLock;
      }
    }
    if (errno == EINTR) continue;
    if (errno == EINVAL || errno == ENOTSUP) break;
    auto status = StatusFromOsError(errno, "Failed to lock file");
    return std::move(tspan).EndWithStatus(std::move(status));
  }
#endif
  while (true) {
    {
      PotentiallyBlockingRegion region;
      if (::flock(fd, LOCK_EX) != -1) {
        return UnlockFlockLock;
      }
      if (errno == EINTR) continue;
      auto status = StatusFromOsError(errno, "Failed to lock file");
      return std::move(tspan).EndWithStatus(std::move(status));
    }
  }
  ABSL_UNREACHABLE();
}

Result<UniqueFileDescriptor> OpenFileWrapper(const std::string& path,
                                             OpenFlags flags) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  FileDescriptor fd = FileDescriptorTraits::Invalid();
  const auto attempt_open = [&] {
    PotentiallyBlockingRegion region;
    fd = ::open(path.c_str(), static_cast<int>(flags) | O_CLOEXEC, 0660);
  };
#ifndef __APPLE__
  attempt_open();
#else
  // Maximum number of attempts to open the file.  On macOS, `open` may
  // return `ENOENT` or `EPERM` if an existing file is deleted concurrently
  // with the call to `open`.  Mitigate this race condition by retrying,
  // limiting the number of retries to avoid getting stuck in an infinite
  // loop if the error is due to a parent directory having been deleted.
  constexpr int kMaxAttempts = 10;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    attempt_open();
    if (fd != FileDescriptorTraits::Invalid() ||
        (errno != ENOENT && errno != EPERM))
      break;
  }
#endif

  if (fd == FileDescriptorTraits::Invalid()) {
    auto status =
        StatusFromOsError(errno, "Failed to open: ", QuoteString(path));
    return std::move(tspan).EndWithStatus(std::move(status));
  } else {
    tspan.Log("fd", fd);
    return UniqueFileDescriptor(fd);
  }
}

Result<ptrdiff_t> ReadFromFile(FileDescriptor fd, void* buf, size_t count,
                               int64_t offset) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"fd", fd}, {"count", count}, {"offset", offset}});
  ssize_t n;
  do {
    PotentiallyBlockingRegion region;
    n = ::pread(fd, buf, count, static_cast<off_t>(offset));
  } while ((n < 0) && (errno == EINTR || errno == EAGAIN));
  if (n >= 0) {
    return n;
  }
  auto status = StatusFromOsError(errno, "Failed to read from file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<ptrdiff_t> WriteToFile(FileDescriptor fd, const void* buf,
                              size_t count) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"fd", fd}, {"count", count}});

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
  auto status = StatusFromOsError(errno, "Failed to write to file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<ptrdiff_t> WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"fd", fd}, {"count", value.size()}});

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
  auto status = StatusFromOsError(errno, "Failed to write to file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status TruncateFile(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});
  PotentiallyBlockingRegion region;
  if (::ftruncate(fd, 0) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno, "Failed to truncate file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status RenameOpenFile(FileDescriptor fd, const std::string& old_name,
                            const std::string& new_name) {
  LoggedTraceSpan tspan(
      __func__, detail_logging.Level(1),
      {{"fd", fd}, {"old_name", old_name}, {"new_name", new_name}});

  PotentiallyBlockingRegion region;
  if (::rename(old_name.c_str(), new_name.c_str()) == 0) {
    return absl::OkStatus();
  }
  auto status =
      StatusFromOsError(errno, "Failed to rename fd: ", absl::StrCat(fd), " ",
                        QuoteString(old_name), " to: ", QuoteString(new_name));
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status DeleteOpenFile(FileDescriptor fd, const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"fd", fd}, {"path", path}});

  PotentiallyBlockingRegion region;
  if (::unlink(path.c_str()) == 0) {
    return absl::OkStatus();
  }
  auto status =
      StatusFromOsError(errno, "Failed to delete: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status DeleteFile(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  PotentiallyBlockingRegion region;
  if (::unlink(path.c_str()) == 0) {
    return absl::OkStatus();
  }
  auto status =
      StatusFromOsError(errno, "Failed to delete: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

uint32_t GetDefaultPageSize() {
  static const uint32_t kDefaultPageSize = []() {
    return sysconf(_SC_PAGE_SIZE);
  }();
  return kDefaultPageSize;
}

Result<MappedRegion> MemmapFileReadOnly(FileDescriptor fd, size_t offset,
                                        size_t size) {
#if ABSL_HAVE_MMAP
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"fd", fd}, {"offset", offset}, {"size", size}});

  if (offset > 0 && offset % GetDefaultPageSize() != 0) {
    return absl::InvalidArgumentError(
        "Offset must be a multiple of the default page size.");
  }
  if (size == 0) {
    struct ::stat info;
    if (::fstat(fd, &info) != 0) {
      return std::move(tspan).EndWithStatus(
          StatusFromOsError(errno, "Failed to stat"));
    }

    uint64_t file_size = GetSize(info);
    if (offset + size > file_size) {
      return std::move(tspan).EndWithStatus(absl::OutOfRangeError(
          absl::StrCat("Requested offset ", offset, " + size ", size,
                       " exceeds file size ", file_size)));
    } else if (size == 0) {
      size = file_size - offset;
    }
    if (size == 0) {
      return MappedRegion(nullptr, 0);
    }
  }

  void* address = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (address == MAP_FAILED) {
    return std::move(tspan).EndWithStatus(
        StatusFromOsError(errno, "Failed to mmap"));
  }
  ::madvise(address, size, MADV_WILLNEED);

  mmap_count.Increment();
  mmap_bytes.IncrementBy(size);
  mmap_active.Increment();
  return MappedRegion(static_cast<const char*>(address), size);
#else
  return absl::UnimplementedError("::mmap not supported");
#endif
}

MappedRegion::~MappedRegion() {
  if (data_) {
    if (::munmap(const_cast<char*>(data_), size_) != 0) {
      ABSL_LOG(FATAL) << StatusFromOsError(errno, "Failed to unmap file");
    }
    mmap_active.Decrement();
  }
}

absl::Status FsyncFile(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});
  PotentiallyBlockingRegion region;
  if (::fsync(fd) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno, "Failed to fsync file");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  PotentiallyBlockingRegion region;
  if (::fstat(fd, info) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno, "Failed to get file info");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  PotentiallyBlockingRegion region;
  if (::stat(path.c_str(), info) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno);
  return std::move(tspan).EndWithStatus(std::move(status));
}

Result<UniqueFileDescriptor> OpenDirectoryDescriptor(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});
  FileDescriptor fd;
  {
    PotentiallyBlockingRegion region;
    fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
  }
  if (fd == FileDescriptorTraits::Invalid()) {
    auto status = StatusFromOsError(
        errno, "Failed to open directory: ", QuoteString(path));
    return std::move(tspan).EndWithStatus(std::move(status));
  }
  return UniqueFileDescriptor(fd);
}

absl::Status MakeDirectory(const std::string& path) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});
  PotentiallyBlockingRegion region;
  if (::mkdir(path.c_str(), 0777) == 0 || errno == EEXIST) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(
      errno, "Failed to create directory: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status FsyncDirectory(FileDescriptor fd) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  PotentiallyBlockingRegion region;
  if (::fsync(fd) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno, "Failed to fsync directory");
  return std::move(tspan).EndWithStatus(std::move(status));
}

}  // namespace internal_os
}  // namespace tensorstore
