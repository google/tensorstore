// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_THREAD_CIRCULAR_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_CIRCULAR_QUEUE_H_

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>

#include "absl/log/absl_check.h"
#include "tensorstore/internal/attributes.h"

namespace tensorstore {
namespace internal_container {

/// CircularQueue implements a simple fifo queue similar in concept
/// to std::deque, however there is no indexed access, only access to either
/// end of the queue.  The implementation uses a single block.
///
/// Methods:
///   size_t capacity() const
///   size_t size() const
///   bool empty() const
///   T& front()
///   T& back()
///   T& operator[]
///   void push_back(T)
///   T& emplace_back(Args...)
///   void pop_front()
///   void clear()
///
template <typename T, typename Allocator = std::allocator<T>>
class CircularQueue {
  using AllocatorTraits = std::allocator_traits<Allocator>;
  using Storage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
  static_assert(sizeof(T) == sizeof(Storage));
  using StorageAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<Storage>;
  using StorageAllocatorTraits = std::allocator_traits<StorageAllocator>;

 public:
  explicit CircularQueue(size_t n)
      : begin_(0), end_(0), mask_(0), buffer_(nullptr) {
    ABSL_CHECK_EQ(n & (n - 1), 0);
    internal_resize(n);
  }

  ~CircularQueue() {
    clear();
    if (buffer_) {
      StorageAllocator storage_alloc(allocator_);
      StorageAllocatorTraits::deallocate(
          storage_alloc, reinterpret_cast<Storage*>(buffer_), mask_ + 1);
    }
  }

  CircularQueue(const CircularQueue&) = delete;
  CircularQueue& operator=(const CircularQueue&) = delete;

  /// Returns the capacity of items in the CircularQueue.
  size_t capacity() const { return mask_ + 1; }

  /// Returns the count of items in the CircularQueue.
  size_t size() const { return end_ - begin_; }

  /// Returns whether the CircularQueue is empty.
  bool empty() const { return !size(); }

  /// Returns the first element.
  T& front() {
    ABSL_CHECK(!empty());
    return buffer_[begin_ & mask_];
  }
  const T& front() const {
    ABSL_CHECK(!empty());
    return buffer_[begin_ & mask_];
  }

  /// Returns the last element.
  T& back() {
    ABSL_CHECK(!empty());
    return buffer_[(end_ - 1) & mask_];
  }
  const T& back() const {
    ABSL_CHECK(!empty());
    return buffer_[(end_ - 1) & mask_];
  }

  /// Access the i-th element.
  T& operator[](size_t i) {
    ABSL_CHECK_LT(i, size());
    return buffer_[(begin_ + i) & mask_];
  }
  const T& operator[](size_t i) const {
    ABSL_CHECK_LT(i, size());
    return buffer_[(begin_ + i) & mask_];
  }

  // Add an element to the back.
  void push_back(const T& val) { emplace_back(val); }
  void push_back(T&& val) { emplace_back(std::move(val)); }

  template <typename... A>
  T& emplace_back(A&&... args) {
    auto* storage = emplace_back_raw();
    AllocatorTraits::construct(allocator_, storage, std::forward<A>(args)...);
    return *storage;
  }

  /// Remove the front element.
  void pop_front() {
    ABSL_CHECK(!empty());
    AllocatorTraits::destroy(allocator_, buffer_ + (begin_++ & mask_));
  }

  /// Clears the container.
  void clear() {
    for (size_t i = begin_; i < end_; i++) {
      AllocatorTraits::destroy(allocator_, buffer_ + (i & mask_));
    }
    begin_ = 0;
    end_ = 0;
  }

 private:
  T* emplace_back_raw() {
    if (size() == capacity()) {
      // Allocate a replacement block.
      internal_resize((mask_ + 1) * 2);
    }
    return buffer_ + (end_++ & mask_);
  }

  // Resizes the backing array to c elements.
  void internal_resize(size_t c) {
    ABSL_CHECK_EQ(c & (c - 1), 0);
    ABSL_CHECK_GT(c, mask_ + 1);
    StorageAllocator storage_alloc(allocator_);

    T* new_buffer = std::launder(reinterpret_cast<T*>(
        StorageAllocatorTraits::allocate(storage_alloc, c)));
    size_t j = 0;
    for (size_t i = begin_; i < end_; i++) {
      auto* storage = buffer_ + (i & mask_);
      AllocatorTraits::construct(allocator_, new_buffer + j++,
                                 std::move(*storage));
      AllocatorTraits::destroy(allocator_, storage);
    }
    if (buffer_) {
      StorageAllocatorTraits::deallocate(
          storage_alloc, reinterpret_cast<Storage*>(buffer_), mask_ + 1);
    }
    begin_ = 0;
    end_ = j;
    mask_ = c - 1;
    buffer_ = new_buffer;
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  size_t begin_;
  size_t end_;
  size_t mask_;
  T* buffer_;
};

}  // namespace internal_container
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_CIRCULAR_QUEUE_H_
