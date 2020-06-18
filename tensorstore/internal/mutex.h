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

namespace internal {
struct ReaderMutexTraits {
  template <typename MutexType>
  static void lock(MutexType& m) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    m.ReaderLock();
  }
  template <typename MutexType>
  static void unlock(MutexType& m) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    m.ReaderUnlock();
  }
};

struct WriterMutexTraits {
  template <typename MutexType>
  static void lock(MutexType& m) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    m.WriterLock();
  }
  template <typename MutexType>
  static void unlock(MutexType& m) ABSL_NO_THREAD_SAFETY_ANALYSIS {
    m.WriterUnlock();
  }
};

template <typename MutexType, typename Traits>
class UniqueLockImpl {
 public:
  using mutex_type = MutexType;
  UniqueLockImpl() = default;
  explicit UniqueLockImpl(mutex_type& m) : mutex_(&m) { Traits::lock(m); }
  UniqueLockImpl(mutex_type& m, std::adopt_lock_t) : mutex_(&m) {}

  template <typename OtherMutexType,
            typename = std::enable_if_t<
                std::is_convertible_v<OtherMutexType&, mutex_type&>>>
  UniqueLockImpl(UniqueLockImpl<OtherMutexType, Traits>&& other)
      : mutex_(other.release()) {}

  void unlock() { mutex_.reset(); }
  explicit operator bool() const { return static_cast<bool>(mutex_); }
  mutex_type* release() { return mutex_.release(); }
  mutex_type* mutex() const { return mutex_.get(); }

 private:
  struct Deleter {
    void operator()(mutex_type* m) const { Traits::unlock(*m); }
  };
  std::unique_ptr<mutex_type, Deleter> mutex_;
};

}  // namespace internal

// Note: UniqueWriterLock and UniqueReaderLock inherit from `UniqueLockImpl`,
// rather than being template aliases, in order to support class template
// argument deduction (CTAD).  C++17 does not support CTAD for template aliases;
// C++20 allows it, though.

/// Unique lock type like `std::unique_lock`, but calls `WriterLock` and
/// `WriterUnlock` in place of `lock` and `unlock` for compatibility with
/// `absl::Mutex`.
template <typename MutexType>
class UniqueWriterLock
    : public internal::UniqueLockImpl<MutexType, internal::WriterMutexTraits> {
  using Base = internal::UniqueLockImpl<MutexType, internal::WriterMutexTraits>;

 private:
  using Base::Base;
};

template <typename MutexType>
explicit UniqueWriterLock(MutexType&) -> UniqueWriterLock<MutexType>;
template <typename MutexType>
UniqueWriterLock(MutexType&, std::adopt_lock_t) -> UniqueWriterLock<MutexType>;

/// Unique lock type like `std::unique_lock`, but calls `ReaderLock` and
/// `ReaderUnlock` in place of `lock` and `unlock` for compatibility with
/// `absl::Mutex`.
template <typename MutexType>
class UniqueReaderLock
    : public internal::UniqueLockImpl<MutexType, internal::ReaderMutexTraits> {
  using Base = internal::UniqueLockImpl<MutexType, internal::ReaderMutexTraits>;

 private:
  using Base::Base;
};

template <typename MutexType>
explicit UniqueReaderLock(MutexType&) -> UniqueReaderLock<MutexType>;
template <typename MutexType>
UniqueReaderLock(MutexType&, std::adopt_lock_t) -> UniqueReaderLock<MutexType>;

namespace internal {

template <typename MutexType, typename Traits>
class ScopedUnlockImpl {
 public:
  using mutex_type = MutexType;
  explicit ScopedUnlockImpl(mutex_type& m) : mutex_(m) { Traits::unlock(m); }
  ~ScopedUnlockImpl() { Traits::lock(mutex_); }

 private:
  mutex_type& mutex_;
};

// Note: ScopedWriterUnlock and ScopedReaderLock inherit from
// `ScopedUnlockImpl`, rather than being template aliases, in order to support
// class template argument deduction (CTAD).  C++17 does not support CTAD for
// template aliases; C++20 allows it, though.

template <typename MutexType>
class ScopedWriterUnlock
    : public ScopedUnlockImpl<MutexType, WriterMutexTraits> {
  using Base = ScopedUnlockImpl<MutexType, WriterMutexTraits>;

 public:
  using Base::Base;
};
template <typename MutexType>
explicit ScopedWriterUnlock(MutexType&) -> ScopedWriterUnlock<MutexType>;

template <typename MutexType>
class ScopedReaderUnlock
    : public ScopedUnlockImpl<MutexType, ReaderMutexTraits> {
  using Base = ScopedUnlockImpl<MutexType, ReaderMutexTraits>;

 public:
  using Base::Base;
};
template <typename MutexType>
explicit ScopedReaderUnlock(MutexType&) -> ScopedReaderUnlock<MutexType>;

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_MUTEX_H_
