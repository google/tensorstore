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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_LOCK_H_
#define TENSORSTORE_INTERNAL_OS_FILE_LOCK_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

class FileLock;

/// Opens a file with an OS write lock.
///
/// This uses os-level advisory file locking to ensure mutual exclusion, and
/// will block until the lock is acquired.
///
/// NOTE: Stale lock files may be left behind if the process crashes; this
/// is acceptable as os-level advisory locks are dropped on process exit.
Result<FileLock> AcquireFileLock(std::string lock_path);

/// Opens a file in exclusive mode to use as a lock.
///
/// If the file already exists then backoff and retry until `timeout` expires.
///
/// NOTE: Stale lock files may be left behind if the process crashes; this
/// will cause the write to fail and will need to be cleaned up manually.
Result<FileLock> AcquireExclusiveFile(std::string lock_path,
                                      absl::Duration timeout);
/// FileLock wraps a lock file.
/// Caller must call `Delete()` or `Close()` before FileLock is destroyed.
class FileLock {
  struct private_t {
    explicit private_t() = default;
  };

 public:
  FileLock(private_t, std::string lock_path, FileDescriptor fd,
           std::optional<internal_os::UnlockFn> unlock_fn)
      : lock_path_(std::move(lock_path)), fd_(fd), unlock_fn_(unlock_fn) {}

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

  FileLock(FileLock&& o) { *this = std::move(o); };
  FileLock& operator=(FileLock&& o) {
    lock_path_ = std::move(o.lock_path_);
    fd_ = std::exchange(o.fd_, FileDescriptorTraits::Invalid());
    unlock_fn_ = std::move(o.unlock_fn_);
    o.unlock_fn_ = std::nullopt;
    return *this;
  };

  ~FileLock();

  FileDescriptor fd() const { return fd_; }
  const std::string& lock_path() const { return lock_path_; }

  // Deletes the lock file and unlocks the file.
  absl::Status Delete() &&;

  // Closes the file descriptor and unlocks the file.
  void Close() &&;

 private:
  friend Result<FileLock> AcquireFileLock(std::string);
  friend Result<FileLock> AcquireExclusiveFile(std::string, absl::Duration);

  inline void Unlock(FileDescriptor fd) {
    if (unlock_fn_) {
      (*std::move(unlock_fn_))(fd);
      unlock_fn_ = std::nullopt;
    }
  }

  std::string lock_path_;
  FileDescriptor fd_;
  std::optional<internal_os::UnlockFn> unlock_fn_;
};

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_LOCK_H_
