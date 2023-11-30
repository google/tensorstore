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

#ifndef TENSORSTORE_INTERNAL_THREAD_BLOCK_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_BLOCK_QUEUE_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <memory>

#include "absl/log/absl_check.h"
#include "tensorstore/internal/attributes.h"

namespace tensorstore {
namespace internal_container {

/// BlockQueue implements a simple fifo queue similar in concept
/// to std::deque, however there is no indexed access, only access to either
/// end of the queue.  The implementation uses a list of blocks; the
/// initial block is sized to hold approximately kMin elements, with
/// each subsequent block doubling in size to a maximum of kMax elements.
///
/// Methods:
///   size_t size() const
///   bool empty() const
///   T& back()
///   T& front()
///   void push_back(T)
///   T& emplace_back(Args...)
///   void pop_front()
///   void clear()
///
template <typename T, size_t kMin = 1024, size_t kMax = 1024,
          typename Allocator = std::allocator<T>>
class BlockQueue;

// SQBlock is a buffer used as a backing SimpleDeque.
template <typename T, typename Allocator = std::allocator<T>>
struct SQBlock {
 private:
  using BlockAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<SQBlock>;
  using ByteAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>;

  constexpr static ptrdiff_t start_offset() {
    struct X {
      SQBlock array;
      T item[1];
    };
    return offsetof(X, item);
  }
  constexpr static size_t start_items() {
    return (start_offset() + sizeof(T) - 1) / sizeof(T);
  }

  // Makes the constructor private; use the New / Delete methods.
  struct private_t {};

 public:
  static SQBlock* New(int64_t c, Allocator* alloc) {
    size_t allocation_bytes =
        (c <= start_items() + 2)
            ? (start_offset() + 2 * sizeof(T))
            : (c -= start_items(), ((c + start_items()) * sizeof(T)));

    ByteAllocator byte_alloc(*alloc);
    void* mem = std::allocator_traits<ByteAllocator>::allocate(
        byte_alloc, allocation_bytes);
    auto* as_array = static_cast<SQBlock*>(mem);
    BlockAllocator array_alloc(*alloc);
    std::allocator_traits<BlockAllocator>::construct(array_alloc, as_array,
                                                     private_t{}, c);
    return as_array;
  }

  static void Delete(SQBlock* ptr, Allocator* alloc) {
    // Skip trivial destructor.
    const size_t allocation_bytes =
        (ptr->capacity() == 2) ? (start_offset() + 2 * sizeof(T))
                               : (start_items() + ptr->capacity()) * sizeof(T);
    BlockAllocator block_alloc(*alloc);
    std::allocator_traits<BlockAllocator>::destroy(block_alloc, ptr);
    void* mem = ptr;
    ByteAllocator byte_alloc(*alloc);
    std::allocator_traits<ByteAllocator>::deallocate(
        byte_alloc, static_cast<char*>(mem), allocation_bytes);
  }

  SQBlock(private_t, size_t c) : end_(begin() + c), next_(nullptr) {}

  SQBlock* next() const { return next_; }
  void set_next(SQBlock* b) { next_ = b; }

  T* begin() {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(this) + start_offset());
  }
  T* end() { return end_; }
  size_t capacity() { return end() - begin(); }

 private:
  T* end_;
  SQBlock* next_;
};

template <typename T, size_t kMin, size_t kMax, typename Allocator>
class BlockQueue {
  using Block = SQBlock<T, Allocator>;
  using AllocatorTraits = std::allocator_traits<Allocator>;

  static_assert(kMin > 0);
  static_assert(kMin <= kMax);

  struct Cursor {
    Cursor(Block* b) : block(b), ptr(b->begin()), end(b->end()) {}
    Cursor() : block(nullptr), ptr(nullptr), end(nullptr) {}

    Block* block;
    T* ptr;
    T* end;
  };

 public:
  BlockQueue() : head_(), tail_(), size_(0) {}

  ~BlockQueue() {
    Block* b = head_.block;
    while (b) {
      Block* next = b->next();
      ClearBlock(b);
      Block::Delete(b, &allocator_);
      b = next;
    }
  }

  BlockQueue(const BlockQueue&) = delete;
  BlockQueue& operator=(const BlockQueue&) = delete;

  /// Returns the count of items in the BlockQueue.
  size_t size() const { return size_; }

  /// Returns whether the BlockQueue is empty.
  bool empty() const { return !size(); }

  /// Returns the first element.
  T& front() {
    ABSL_CHECK(!empty());
    return *head_.ptr;
  }
  const T& front() const {
    ABSL_CHECK(!empty());
    return *head_.ptr;
  }

  /// Returns the last element.
  T& back() {
    ABSL_CHECK(!empty());
    return *((tail_.ptr) - 1);
  }
  const T& back() const {
    ABSL_CHECK(!empty());
    return *((tail_.ptr) - 1);
  }

  /// Add an element to the back.
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
    ABSL_CHECK(head_.block);
    AllocatorTraits::destroy(allocator_, head_.ptr);
    ++head_.ptr;
    --size_;
    if (empty()) {
      // Reset head and tail to the start of the (only) block.
      ABSL_CHECK_EQ(head_.block, tail_.block);
      head_.ptr = tail_.ptr = head_.block->begin();
      return;
    }
    if (head_.ptr == head_.end) {
      Block* n = head_.block->next();
      Block::Delete(head_.block, &allocator_);
      head_ = Cursor(n);
    }
  }

  /// Clears the container.
  void clear() {
    Block* b = head_.block;
    if (!b) {
      ABSL_CHECK(empty());
      return;
    }
    while (b) {
      Block* next = b->next();
      ClearBlock(b);
      if (head_.block != b) {
        Block::Delete(b, &allocator_);
      }
      b = next;
    }
    b = head_.block;
    b->set_next(nullptr);
    head_ = tail_ = Cursor(b);
    size_ = 0;
  }

 private:
  T* emplace_back_raw() {
    if (tail_.ptr == tail_.end) {
      // Allocate a new block.
      size_t capacity = kMin;
      if (tail_.block) {
        capacity = 2 * tail_.block->capacity();
        if (capacity > kMax) capacity = kMax;
      }

      auto* b = Block::New(capacity, &allocator_);
      if (!head_.block) {
        ABSL_CHECK(tail_.block == nullptr);
        head_ = Cursor(b);
      } else {
        ABSL_CHECK(head_.block != nullptr);
        tail_.block->set_next(b);
      }
      tail_ = Cursor(b);
    }

    ++size_;
    return tail_.ptr++;
  }

  void ClearBlock(Block* b) {
    auto* begin = b == head_.block ? head_.ptr : b->begin();
    auto* end = b == tail_.block ? tail_.ptr : b->end();
    for (; begin != end; ++begin) {
      AllocatorTraits::destroy(allocator_, begin);
    }
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  Cursor head_;
  Cursor tail_;
  size_t size_;
};

}  // namespace internal_container
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_BLOCK_QUEUE_H_
