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

#ifndef TENSORSTORE_INTERNAL_THREAD_SINGLE_PRODUCER_QUEUE_H_
#define TENSORSTORE_INTERNAL_THREAD_SINGLE_PRODUCER_QUEUE_H_

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "tensorstore/internal/attributes.h"

namespace tensorstore {
namespace internal_container {

/// SingleProducerQueue is a lock-free single-producer multi-consumer FIFO queue
/// See: https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf
/// and https://fzn.fr/readings/ppopp13.pdf
///
/// The queue elements, T, must be trivially_destructible and copyable.
/// The queue allows a single owner/writer thread to produce and consume
/// elements using the push and try_pop primitives.
/// Other threads can `steal` items using try_steal/try_steal_n primitives.
///
/// When kCanResize is false, the buffer will fail push operations when full,
/// otherwise the buffer will be resized to double the current size without
/// deallocation.
///
/// Methods:
///   int64_t capacity() const
///   size_t size() const
///   bool empty() const
///   bool push(T)
///   optional_t try_pop()
///   optional_t try_steal()
///
template <typename T, bool kCanResize = true,
          typename Allocator = std::allocator<T>>
class SingleProducerQueue;

// SPQArray is a circular buffer used as a backing store for
// SingleProducerQueue.
template <typename T, typename Allocator = std::allocator<T>>
struct SPQArray {
 private:
  static_assert(std::is_trivially_destructible_v<T>);

  using ArrayAllocator = typename std::allocator_traits<
      Allocator>::template rebind_alloc<SPQArray>;
  using ByteAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>;

  constexpr static ptrdiff_t start_offset() {
    struct X {
      SPQArray array;
      std::atomic<T> item[1];
    };
    return offsetof(X, item);
  }

  constexpr static size_t alloc_size(int64_t c) {
    struct X {
      SPQArray array;
      std::atomic<T> item[1];
    };
    return sizeof(X) + (c - 1) * sizeof(std::atomic<T>);
  }

  // Makes the constructor private; use the New / Delete methods.
  struct private_t {};

 public:
  static SPQArray* New(int64_t c, SPQArray* retired, Allocator* alloc) {
    size_t allocation_bytes = alloc_size(c);

    ByteAllocator byte_alloc(*alloc);
    void* mem = std::allocator_traits<ByteAllocator>::allocate(
        byte_alloc, allocation_bytes);
    auto* as_array = static_cast<SPQArray*>(mem);
    ArrayAllocator array_alloc(*alloc);
    std::allocator_traits<ArrayAllocator>::construct(array_alloc, as_array,
                                                     private_t{}, c, retired);
    return as_array;
  }

  static void Delete(SPQArray* ptr, Allocator* alloc) {
    // Skip trivial destructor.
    const size_t allocation_bytes = alloc_size(ptr->capacity);
    void* mem = ptr;
    ByteAllocator byte_alloc(*alloc);
    std::allocator_traits<ByteAllocator>::deallocate(
        byte_alloc, static_cast<char*>(mem), allocation_bytes);
  }

  SPQArray(private_t, int64_t c, SPQArray* retired)
      : capacity(c), mask(c - 1), retired(retired) {}

  SPQArray* resize(int64_t b, int64_t t, Allocator* alloc) {
    auto* a = SPQArray::New(2 * capacity, this, alloc);
    for (int64_t i = t; i != b; ++i) {
      a->item(i).store(  //
          item(i).load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return a;
  }

  std::atomic<T>* buffer() {
    return reinterpret_cast<std::atomic<T>*>(reinterpret_cast<char*>(this) +
                                             start_offset());
  }

  std::atomic<T>& item(int64_t i) { return buffer()[i & mask]; }

  int64_t capacity;
  int64_t mask;
  SPQArray* retired;
};

template <typename T, bool kCanResize, typename Allocator>
class SingleProducerQueue {
  static_assert(std::is_trivially_destructible_v<T>);

  std::nullopt_t missing(std::false_type) { return std::nullopt; }
  std::nullptr_t missing(std::true_type) { return nullptr; }

  using Array = SPQArray<T, Allocator>;

 public:
  using optional_t =
      std::conditional_t<std::is_pointer_v<T>, T, std::optional<T>>;

  SingleProducerQueue(int64_t n)
      : top_(0), bottom_(0), array_(Array::New(n, nullptr, &allocator_)) {
    ABSL_CHECK_EQ(n & (n - 1), 0);
  }

  ~SingleProducerQueue() {
    // Avoid recursion on destruction.
    Array* a = array_.load(std::memory_order_relaxed);
    while (a) {
      Array* b = a->retired;
      a->retired = nullptr;
      Array::Delete(a, &allocator_);
      a = b;
    }
  }

  /// Returns the capacity of the TaskQueue.
  int64_t capacity() const {
    return array_.load(std::memory_order_relaxed)->capacity;
  }

  /// Returns the count of items in the TaskQueue.
  size_t size() const {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return static_cast<size_t>(b > t ? b - t : 0);
  }

  /// Returns whether the TaskQueue is empty.
  bool empty() const { return !size(); }

  /// Attempt to add an an item to the queue.
  /// Returns false when the item cannot be added.
  /// May be called by the owning thread.
  bool push(T x) {
    auto b = bottom_.load(std::memory_order_relaxed);
    auto t = top_.load(std::memory_order_acquire);
    Array* a = array_.load(std::memory_order_relaxed);

    if (a->capacity < (b - t) + 1) {
      // Full, resize.  Consider Chase-Lev 4.1 to reclaim buffer space.
      if (!kCanResize) return false;
      a = a->resize(b, t, &allocator_);
      array_.store(a, std::memory_order_release);
    }
    a->item(b).store(std::move(x), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  /// Remove an item from the queue.
  /// May be called by the owning thread.
  optional_t try_pop() {
    auto b = bottom_.load(std::memory_order_relaxed) - 1;
    Array* a = array_.load(std::memory_order_relaxed);
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto t = top_.load(std::memory_order_relaxed);

    if (t > b) {
      // Empty.
      bottom_.store(b + 1, std::memory_order_relaxed);
      return missing(std::is_pointer<T>{});
    }

    // Non-empty
    if (t == b) {
      // Single element
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
        // Failed race.
        bottom_.store(b + 1, std::memory_order_relaxed);
        return missing(std::is_pointer<T>{});
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
    }

    return a->item(b).load(std::memory_order_relaxed);
  }

  /// Attempt to steal an item from this queue.
  /// May be called by non-owning threads.
  optional_t try_steal() {
    auto t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto b = bottom_.load(std::memory_order_acquire);

    if (t >= b) {
      return missing(std::is_pointer<T>{});
    }

    Array* a = array_.load(std::memory_order_consume);
    T x = a->item(t).load(std::memory_order_relaxed);
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
      return missing(std::is_pointer<T>{});
    }
    return x;
  }

 private:
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Allocator allocator_;
  ABSL_CACHELINE_ALIGNED std::atomic<int64_t> top_;
  std::atomic<int64_t> bottom_;
  std::atomic<Array*> array_;
};

}  // namespace internal_container
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_SINGLE_PRODUCER_QUEUE_H_
