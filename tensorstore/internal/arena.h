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

#ifndef TENSORSTORE_INTERNAL_ARENA_H_
#define TENSORSTORE_INTERNAL_ARENA_H_

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "tensorstore/internal/exception_macros.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Simple memory allocation arena implementation.
///
/// Allocations are served from the optional fixed-size buffer provided to the
/// constructor (typically stack allocated) while it has available capacity.
/// Allocations that do not fit in the fixed size buffer are handled using
/// `::operator new`.
///
/// Memory allocated from the fixed-size buffer is not reclaimed until the
/// `Arena` is destroyed (`deallocate` is no-op), but `deallocate` still must be
/// called for every allocation in order to handle allocations that did not fit
/// in the fixed-size buffer.
class Arena {
 public:
  Arena() : remaining_bytes_(0) {}
  explicit Arena(span<unsigned char> initial_buffer)
      : initial_buffer_(initial_buffer),
        remaining_bytes_(initial_buffer.size()) {}

  /// Allocates memory suitable for `n` objects of type `T`.
  ///
  /// Does not call any constructors of `T`.
  ///
  /// \param n Number of objects to allocate.
  /// \param alignment Optional.  Specifies an alignment to use other than the
  ///     default of `alignof(T)`.
  template <typename T = unsigned char>
  T* allocate(std::size_t n, std::size_t alignment = alignof(T)) {
    std::size_t num_bytes;
    if (MulOverflow(n, sizeof(T), &num_bytes)) {
      TENSORSTORE_THROW_BAD_ALLOC;
    }
    void* ptr = static_cast<void*>(initial_buffer_.end() - remaining_bytes_);
    if (std::align(alignment, num_bytes, ptr, remaining_bytes_)) {
      remaining_bytes_ -= num_bytes;
    } else {
      ptr = ::operator new(num_bytes, std::align_val_t(alignment));
    }
    return static_cast<T*>(ptr);
  }

  /// Deallocates memory returned by `allocate`.
  ///
  /// \tparam T Type argument used with `allocate`.
  /// \param p The pointer returned by `allocate`.
  /// \param n The size parameter passed to `allocate`.
  /// \param alignment The alignment parameter passed to `allocate`.
  /// \remark For memory allocated within the fixed-size buffer, the memory is
  ///     not actually reclaimed until the `Arena` is destroyed.
  template <typename T>
  void deallocate(T* p, std::size_t n, std::size_t alignment = alignof(T)) {
    if (static_cast<void*>(p) >= static_cast<void*>(initial_buffer_.data()) &&
        static_cast<void*>(p + n) <=
            static_cast<void*>(initial_buffer_.data() +
                               initial_buffer_.size())) {
      return;
    }
    ::operator delete(static_cast<void*>(p), n * sizeof(T),
                      std::align_val_t(alignment));
  }

 private:
  span<unsigned char> initial_buffer_;
  size_t remaining_bytes_;
};

/// C++ standard library Allocator implementation that uses `Arena`.
///
/// This class simply holds a pointer to an `Arena`, which must outlive the
/// allocator.
///
/// Example usage:
///
///     unsigned char buffer[32 * 1024];
///     Arena arena(buffer);
///     std::vector<int, ArenaAllocator<int>> vec(100, &arena);
template <typename T = unsigned char>
class ArenaAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using void_pointer = void*;
  using const_void_pointer = const void*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  template <typename U>
  struct rebind {
    using other = ArenaAllocator<U>;
  };

  /// Constructs from the specified arena.
  ///
  /// \param arena Non-null pointer to arena.
  ArenaAllocator(Arena* arena) : arena_(arena) {}

  template <typename U>
  ArenaAllocator(ArenaAllocator<U> other) : arena_(other.arena()) {}

  T* allocate(std::size_t n) const { return arena_->allocate<T>(n); }

  void deallocate(T* p, std::size_t n) const { arena_->deallocate(p, n); }

  template <typename... Arg>
  void construct(T* p, Arg&&... arg) {
    new (p) T(std::forward<Arg>(arg)...);
  }

  void destroy(T* p) { p->~T(); }

  /// Returns the associated arena.  Always non-null.
  Arena* arena() const { return arena_; }

  friend bool operator==(ArenaAllocator a, ArenaAllocator b) {
    return a.arena_ == b.arena_;
  }

  friend bool operator!=(ArenaAllocator a, ArenaAllocator b) {
    return a.arena_ != b.arena_;
  }

  Arena* arena_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ARENA_H_
