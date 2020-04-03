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

#ifndef TENSORSTORE_INTERNAL_MUTEX_H_
#define TENSORSTORE_INTERNAL_MUTEX_H_

#include <memory>
#include <mutex>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {

/// \brief Trivial wrapper around absl::Mutex providing C++ standard library
/// compatibility.
///
/// This class satisfies the C++ standard library Lockable concept and can be
/// used with std::unique_lock.
class ABSL_LOCKABLE Mutex : public absl::Mutex {
 public:
  void lock() ABSL_EXCLUSIVE_LOCK_FUNCTION() { this->Lock(); }
  void unlock() ABSL_UNLOCK_FUNCTION() { this->Unlock(); }
  bool try_lock() ABSL_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return this->TryLock();
  }
};

/// `std::unique_lock` equivalent for holding a shared (reader) lock on an
/// `absl::Mutex`.
class UniqueReaderLock {
 public:
  using mutex_type = absl::Mutex;
  UniqueReaderLock() = default;
  explicit UniqueReaderLock(mutex_type& m) ABSL_SHARED_LOCK_FUNCTION(&m)
      : mutex_(&m) {
    mutex_->ReaderLock();
  }
  UniqueReaderLock(mutex_type& m, std::adopt_lock_t) : mutex_(&m) {}

  mutex_type* release() noexcept { return mutex_.release(); }
  mutex_type* mutex() const noexcept { return mutex_.get(); }

  explicit operator bool() const noexcept { return static_cast<bool>(mutex_); }

 private:
  struct Deleter {
    void operator()(absl::Mutex* m) const ABSL_UNLOCK_FUNCTION(m) {
      m->ReaderUnlock();
    }
  };
  std::unique_ptr<absl::Mutex, Deleter> mutex_;
};

namespace internal {

class ScopedMutexUnlock {
 public:
  explicit ScopedMutexUnlock(absl::Mutex* mutex) : mutex_(mutex) {
    mutex->Unlock();
  }
  ~ScopedMutexUnlock() { mutex_->Lock(); }

 private:
  absl::Mutex* mutex_;
};

}  // namespace internal

}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_MUTEX_H_
