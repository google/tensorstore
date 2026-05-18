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

#ifndef TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_
#define TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_

#include <utility>

#include "absl/status/status.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {

class UniqueFileDescriptor;

#ifdef _WIN32
// Specializations for Windows.

/// Representation of open file/directory.
using FileDescriptor = HANDLE;  // HANDLE

static inline FileDescriptor InvalidFileDescriptor() {
  return INVALID_HANDLE_VALUE;
}

#else
// Specializations for Posix.

/// Representation of open file/directory.
using FileDescriptor = int;

static constexpr FileDescriptor InvalidFileDescriptor() { return -1; }

#endif

/// Closes the specified file descriptor.
absl::Status CloseFileDescriptor(FileDescriptor fd);

/// Unique handle to an open file descriptor.
///
/// The file descriptor is closed automatically by the destructor.
class UniqueFileDescriptor {
 public:
  UniqueFileDescriptor() : fd_(InvalidFileDescriptor()) {}
  explicit UniqueFileDescriptor(FileDescriptor fd) : fd_(fd) {}

  UniqueFileDescriptor(const UniqueFileDescriptor&) = delete;
  UniqueFileDescriptor& operator=(const UniqueFileDescriptor&) = delete;

  UniqueFileDescriptor(UniqueFileDescriptor&& other) noexcept
      : fd_(std::exchange(other.fd_, InvalidFileDescriptor())) {}

  UniqueFileDescriptor& operator=(UniqueFileDescriptor&& other) noexcept {
    reset(std::exchange(other.fd_, InvalidFileDescriptor()));
    return *this;
  }

  ~UniqueFileDescriptor() {
    if (valid()) CloseFileDescriptor(fd_).IgnoreError();
  }

  bool valid() const { return fd_ != InvalidFileDescriptor(); }

  void reset(FileDescriptor fd = InvalidFileDescriptor()) {
    if (valid()) CloseFileDescriptor(fd_).IgnoreError();
    fd_ = fd;
  }

  FileDescriptor get() const { return fd_; }

  FileDescriptor release() {
    return std::exchange(fd_, InvalidFileDescriptor());
  }

  absl::Status Close() && {
    if (!valid()) return absl::OkStatus();
    auto status = CloseFileDescriptor(fd_);
    fd_ = InvalidFileDescriptor();
    return status;
  }

 private:
  FileDescriptor fd_;
};

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FILE_DESCRIPTOR_H_
