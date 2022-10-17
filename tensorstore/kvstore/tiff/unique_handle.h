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

#ifndef TENSORSTORE_KVSTORE_FILE_UNIQUE_HANDLE_H_
#define TENSORSTORE_KVSTORE_FILE_UNIQUE_HANDLE_H_

#include <utility>

namespace tensorstore {
namespace internal {

/// Unique handle class, used for file descriptors.
///
/// The behavior is specified by a `Traits` type with the following members:
///
///     static Handle Invalid() noexcept;
///
///     static void Close(Handle handle) noexcept;
///
/// \tparam Handle Handle type.
/// \tparam Traits Specifies the behavior.
template <typename Handle, typename Traits>
class UniqueHandle {
 public:
  UniqueHandle() : fd_(Traits::Invalid()) {}
  explicit UniqueHandle(Handle fd) : fd_(fd) {}
  bool valid() const { return fd_ != Traits::Invalid(); }
  UniqueHandle(UniqueHandle&& other)
      : fd_(std::exchange(other.fd_, Traits::Invalid())) {}
  UniqueHandle& operator=(UniqueHandle&& other) {
    reset(std::exchange(other.fd_, Traits::Invalid()));
    return *this;
  }
  void reset(Handle fd = Traits::Invalid()) {
    if (valid()) Traits::Close(fd_);
    fd_ = fd;
  }
  Handle get() const { return fd_; }
  Handle release() { return std::exchange(fd_, Traits::Invalid()); }
  UniqueHandle(const UniqueHandle&) = delete;
  UniqueHandle& operator=(const UniqueHandle&) = delete;
  ~UniqueHandle() {
    if (valid()) Traits::Close(fd_);
  }

 private:
  Handle fd_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_UNIQUE_HANDLE_H_
