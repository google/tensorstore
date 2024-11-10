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

#include "tensorstore/internal/os/file_lock.h"

#include <stdint.h>

#include <cassert>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {
namespace {

auto& lock_contention = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/file/lock_contention",
    internal_metrics::MetricMetadata(
        "Number of times a file lock is contended"));

}  // namespace

FileLock::~FileLock() {
  assert(fd_ == FileDescriptorTraits::Invalid());
  assert(unlock_fn_ == std::nullopt);
}

absl::Status FileLock::Delete() && {
  assert(fd_ != FileDescriptorTraits::Invalid());

  auto fd = std::exchange(fd_, FileDescriptorTraits::Invalid());
  auto status = internal_os::DeleteOpenFile(fd, lock_path_);
  Unlock(fd);
  FileDescriptorTraits::Close(fd);
  return MaybeAnnotateStatus(std::move(status), "Failed to clean lock file");
}

void FileLock::Close() && {
  assert(fd_ != FileDescriptorTraits::Invalid());
  auto fd = std::exchange(fd_, FileDescriptorTraits::Invalid());
  Unlock(fd);
  FileDescriptorTraits::Close(fd);
}

Result<FileLock> AcquireFileLock(std::string lock_path) {
  using private_t = FileLock::private_t;
  TENSORSTORE_ASSIGN_OR_RETURN(
      UniqueFileDescriptor fd,
      internal_os::OpenFileWrapper(lock_path, OpenFlags::DefaultWrite));
  FileInfo a, b;
  FileInfo* info = &a;

  // Is this a network filesystem?
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd.get(), info));
  if (!internal_os::IsRegularFile(*info)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Not a regular file: ", lock_path));
  }

  // Loop until lock is acquired successfully.
  while (true) {
    // Acquire lock.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto unlock_fn, internal_os::AcquireFdLock(fd.get()),
        MaybeAnnotateStatus(_, absl::StrCat("Failed to acquire lock on file: ",
                                            QuoteString(lock_path))));

    // Reopening the file should give the same value since the lock is held.
    TENSORSTORE_ASSIGN_OR_RETURN(
        UniqueFileDescriptor other_fd,
        internal_os::OpenFileWrapper(lock_path, OpenFlags::DefaultWrite));

    FileInfo* other_info = info == &a ? &b : &a;
    TENSORSTORE_RETURN_IF_ERROR(
        internal_os::GetFileInfo(other_fd.get(), other_info));
    if (internal_os::GetDeviceId(a) == internal_os::GetDeviceId(b) &&
        internal_os::GetFileId(a) == internal_os::GetFileId(b)) {
      // Lock was acquired successfully.
      return FileLock(private_t(), std::move(lock_path), fd.release(),
                      std::move(unlock_fn));
    }

    // Release lock and try again.
    (*std::move(unlock_fn))(fd.get());
    info = other_info;
    fd = std::move(other_fd);
    lock_contention.Increment();
  }
}

/* static */
Result<FileLock> AcquireExclusiveFile(std::string lock_path,
                                      absl::Duration timeout) {
  using private_t = FileLock::private_t;

  FileInfo info;
  auto start = absl::Now();

  // Determine whether the lock file is stale.
  auto detect_stale_lock = [&]() mutable {
    auto read_fd = OpenFileWrapper(lock_path, OpenFlags::OpenReadOnly);
    if (read_fd.ok()) {
      TENSORSTORE_RETURN_IF_ERROR(GetFileInfo(read_fd->get(), &info));
      if (!internal_os::IsRegularFile(info)) {
        // A lock file must be a regular file, not a symlink or directory.
        return absl::FailedPreconditionError(
            absl::StrCat("Not a regular file: ", lock_path));
      }
      if (GetMTime(info) < (start - timeout)) {
        // NOTE: Automatic cleanup of stale lock could be added.
        ABSL_LOG(WARNING) << "Potential stale lock file: " << lock_path
                          << " with mtime: " << GetMTime(info);
      }
    }
    return absl::OkStatus();
  };

  int m = 0;
  int n = 0;
  do {
    absl::SleepFor(m * absl::Milliseconds(1));
    // Cubic backoff.  m = n^3/3 + n^2/2 + (7 n)/6 + 1.
    m += (n * n) + 1;
    n++;
    if (m > 1000) m = 1000;

    auto fd = internal_os::OpenFileWrapper(
        lock_path,
        OpenFlags::Create | OpenFlags::Exclusive | OpenFlags::OpenReadWrite);
    if (fd.ok()) {
      TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd->get(), &info));
      return FileLock(private_t{}, std::move(lock_path), fd->release(),
                      std::nullopt);
    }
    if (!absl::IsAlreadyExists(fd.status())) {
      // Only loop if the file already exists.
      return std::move(fd).status();
    }
    if (n == 1 && timeout > absl::ZeroDuration()) {
      // The first time through the loop, check for a stale lock file and
      // print a warning.
      TENSORSTORE_RETURN_IF_ERROR(detect_stale_lock());
    }
  } while (absl::Now() < start + timeout);

  return absl::DeadlineExceededError(
      absl::StrCat("Failed to open lock file: ", lock_path));
}

}  // namespace internal_os
}  // namespace tensorstore
