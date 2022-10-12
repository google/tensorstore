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

#ifndef _WIN32

#include "tensorstore/kvstore/file/posix_file_util.h"

// More system headers
#include <sys/uio.h>
#include <unistd.h>

#include "absl/container/inlined_vector.h"
#include "tensorstore/internal/os_error_code.h"
#include "tensorstore/kvstore/file/file_util.h"
#include "tensorstore/kvstore/file/potentially_blocking_region.h"
#include "tensorstore/util/quote_string.h"

// Most modern unix allow 1024 iovs.
#if defined(UIO_MAXIOV)
#define TENSORSTORE_MAXIOV UIO_MAXIOV
#else
#define TENSORSTORE_MAXIOV 1024
#endif

using ::tensorstore::internal::PotentiallyBlockingRegion;

namespace tensorstore {
namespace internal_file_util {

UniqueFileDescriptor OpenFileForWriting(const std::string& path) {
  UniqueFileDescriptor fd;
  const auto attempt_open = [&] {
    PotentiallyBlockingRegion region;
    fd.reset(::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0666));
  };
#ifndef __APPLE__
  attempt_open();
#else
  // Maximum number of attempts to open the file.  On macOS, `open` may return
  // `ENOENT` or `EPERM` if an existing file is deleted concurrently with the
  // call to `open`.  We mitigate this race condition by retrying, but we limit
  // the number of retries to avoid getting stuck in an infinite loop if the
  // error is due to a parent directory having been deleted.
  constexpr int kMaxAttempts = 100;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    attempt_open();
    if (fd.valid() || (errno != ENOENT && errno != EPERM)) break;
  }
#endif
  return fd;
}

std::ptrdiff_t WriteCordToFile(FileDescriptor fd, absl::Cord value) {
  absl::InlinedVector<iovec, 16> iovs;

  for (absl::string_view chunk : value.Chunks()) {
    struct iovec iov;
    iov.iov_base = const_cast<char*>(chunk.data());
    iov.iov_len = chunk.size();
    iovs.emplace_back(iov);
    if (iovs.size() >= TENSORSTORE_MAXIOV) break;
  }
  PotentiallyBlockingRegion region;
  std::ptrdiff_t n = ::writev(fd, iovs.data(), iovs.size());
  if (!value.empty() && n == 0) {
    errno = ENOSPC;
  }
  return n;
}

bool FileLockTraits::Acquire(int fd) {
  PotentiallyBlockingRegion region;
  while (true) {
    // This blocks until the lock is acquired (SETLKW).  If any signal is
    // received by the current thread, `fcntl` returns `EINTR`.
    //
    // Note: If we wanted to support cancellation while waiting on the lock,
    // we could pass the Promise to this function and register an
    // ExecuteWhenNotNeeded function that uses `pthread_kill` to send a signal
    // to this thread to abort the `fcntl` call (taking care to do it in a
    // race-free way).
#ifdef __linux__
    // Use Linux 3.15+ open file descriptor lock.
    struct ::flock lock;
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_pid = 0;
    if (::fcntl(fd, 38 /*F_OFD_SETLKW*/, &lock) == 0) return true;
#else
    // Use `flock` on BSD/Mac OS.
    if (::flock(fd, LOCK_EX) == 0) return true;
#endif
    if (errno == EINTR) continue;
    return false;
  }
}

void FileLockTraits::Close(int fd) {
  PotentiallyBlockingRegion region;
  // This releases a lock acquired above.
  // This is not strictly necessary as the posix/linux locks will be released
  // when the fd is closed, but it allows easier reasoning by making locking
  // behave similarly across platforms.
  while (true) {
#ifdef __linux__
    // Use Linux 3.15+ open file descriptor lock.
    struct ::flock lock;
    lock.l_type = F_UNLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_pid = 0;
    if (::fcntl(fd, 37 /*F_OFD_SETLK*/, &lock) != EINTR) return;
#else
    // Use `flock` on BSD/Mac OS.
    if (::flock(fd, LOCK_UN) != EINTR) return;
#endif
  }
}

bool DirectoryIterator::Next() {
  PotentiallyBlockingRegion region;
  e = ::readdir(dir.get());
  return e != nullptr;
}

/// IsDirectoryAt returns whethere the `name` at the given directory fd is a
/// directory or not by using fstatat.
bool IsDirectoryAt(int dir_fd, const char* name) {
  PotentiallyBlockingRegion region;
  struct ::stat statbuf;
  if (::fstatat(dir_fd, name, &statbuf, AT_SYMLINK_NOFOLLOW) != 0) {
    // Error stating file, assume it's not a directory.
    return false;
  }
  return S_ISDIR(statbuf.st_mode);
}

bool DirectoryIterator::is_directory() const {
  const auto d_type = e->d_type;
  return (d_type == DT_DIR || (d_type == DT_UNKNOWN &&
                               IsDirectoryAt(::dirfd(dir.get()), e->d_name)));
}

DirectoryIterator::Entry DirectoryIterator::GetEntry() const {
  PotentiallyBlockingRegion region;
  return {::dirfd(dir.get()), e->d_name};
}

bool DirectoryIterator::Make(Entry entry,
                             std::unique_ptr<DirectoryIterator>* new_iterator) {
  PotentiallyBlockingRegion region;
  UniqueFileDescriptor new_dir(
      ::openat(entry.dir_fd, entry.name,
               O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                   (entry.dir_fd == AT_FDCWD ? 0 : O_NOFOLLOW)));
  if (!new_dir.valid()) {
    switch (errno) {
      case ENOENT:
        return true;
      default:
        return false;
    }
  }
  UniqueDir dir(::fdopendir(new_dir.release()));
  if (!dir) {
    return false;
  }
  auto* it = new DirectoryIterator;
  it->dir = std::move(dir);
  it->e = nullptr;
  new_iterator->reset(it);
  return true;
}

Result<std::string> GetCwd() {
  std::string buf;
  buf.resize(256);
  while (true) {
    if (::getcwd(buf.data(), buf.size()) != nullptr) {
      buf.resize(std::strlen(buf.data()));
      return buf;
    }
    if (errno == ERANGE) {
      buf.resize(buf.size() * 2);
      continue;
    }
    return internal::StatusFromOsError(
        errno, "Failed to determine current working directory");
  }
}

absl::Status SetCwd(const std::string& path) {
  if (::chdir(path.c_str()) != 0) {
    return internal::StatusFromOsError(
        errno, "Failed to set current working directory to: ",
        tensorstore::QuoteString(path));
  }
  return absl::OkStatus();
}

}  // namespace internal_file_util
}  // namespace tensorstore

#endif
