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

#ifndef TENSORSTORE_INTERNAL_LOCK_COLLECTION_H_
#define TENSORSTORE_INTERNAL_LOCK_COLLECTION_H_

#include <cassert>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace internal {

/// Facilitates locking a variable number of locks in a deadlock-free manner.
///
/// This serves a similar purpose as `std::lock` and `std::scoped_lock` but for
/// the case where the number of locks is determined at run-time.
///
/// This satisfies the C++ BasicLockable requirements
/// (https://en.cppreference.com/w/cpp/named_req/BasicLockable) and is
/// compatible with `std::unique_lock` and `std::lock_guard`.
class ABSL_LOCKABLE LockCollection {
 public:
  using LockFunction = void (*)(void* data, bool lock);

  /// Registers `data` with the lock collection but does not acquire a lock.
  /// The lock will be acquired when `lock()` is called.
  ///
  /// If the same `data` pointer is registered more than once, only a single
  /// request will be used.  A request with `shared=false` takes precedence over
  /// a request with `shared=true`.
  ///
  /// This must not be called while this lock collection is locked.
  ///
  /// \param data The object to which the lock applies.  Must have an alignment
  ///     of at least 2.
  /// \param lock_function The function to invoke to acquire or release the lock
  ///     on `data`.  If a given combination of `data` and `shared` is
  ///     registered more than once, the same `lock_function` should be
  ///     specified in all registrations.
  /// \param shared Specifies whether `lock_function` acquires a "shared" rather
  ///     than "exclusive" lock.  In the case of multiple requests with the same
  ///     `data` pointer, requests with `shared=true` have lower precedence than
  ///     requests with `shared=false`.
  void Register(void* data, LockFunction lock_function, bool shared)
      ABSL_LOCKS_EXCLUDED(this) {
    assert(data);
    locks_.emplace_back(data, lock_function, shared);
  }

  template <typename BasicLockable>
  void Register(BasicLockable& lockable) ABSL_LOCKS_EXCLUDED(this) {
    static_assert(alignof(BasicLockable) >= 2);
    locks_.emplace_back(&lockable, &BasicLockableLockFunction<BasicLockable>,
                        /*shared=*/false);
  }

  /// Convenience interface for registering a shared lock on `mutex`.
  void RegisterShared(absl::Mutex& mutex) ABSL_LOCKS_EXCLUDED(this) {
    static_assert(alignof(absl::Mutex) >= 2);
    Register(&mutex, &MutexSharedLockFunction, /*shared=*/true);
  }

  /// Convenience interface for registering an exclusive lock on `mutex`.
  void RegisterExclusive(absl::Mutex& mutex) ABSL_LOCKS_EXCLUDED(this) {
    static_assert(alignof(absl::Mutex) >= 2);
    Register(&mutex, &MutexExclusiveLockFunction, /*shared=*/false);
  }

  /// Acquires all of the locks that were registered, in order of their
  /// pointers.
  void lock() ABSL_EXCLUSIVE_LOCK_FUNCTION();

  /// Releases all of the locks that were registered.
  void unlock() ABSL_UNLOCK_FUNCTION();

 private:
  constexpr static std::uintptr_t kTagMask = 1;
  constexpr static std::uintptr_t kDataPointerMask = ~kTagMask;

  static void MutexSharedLockFunction(void* mutex, bool lock);
  static void MutexExclusiveLockFunction(void* mutex, bool lock);

  template <typename BasicLockable>
  static void BasicLockableLockFunction(void* data, bool lock) {
    auto* mutex = static_cast<BasicLockable*>(data);
    if (lock) {
      mutex->lock();
    } else {
      mutex->unlock();
    }
  }

  struct Entry {
    explicit Entry(void* data, LockFunction lock_function, bool shared) {
      tagged_pointer = reinterpret_cast<std::uintptr_t>(data);
      assert(!(tagged_pointer & kTagMask));
      tagged_pointer |= static_cast<std::uintptr_t>(shared);
      this->lock_function = lock_function;
    }

    void* data() const {
      return reinterpret_cast<void*>(tagged_pointer & kDataPointerMask);
    }

    std::uintptr_t tagged_pointer;
    LockFunction lock_function;
  };
  absl::InlinedVector<Entry, 4> locks_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_LOCK_COLLECTION_H_
