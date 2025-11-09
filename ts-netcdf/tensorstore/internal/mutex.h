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

#include "absl/synchronization/mutex.h"

#if defined(__clang__)
#define TENSORSTORE_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define TENSORSTORE_THREAD_ANNOTATION_ATTRIBUTE(x)  // no-op
#endif

namespace tensorstore {
namespace internal {

#if !defined(NDEBUG)
template <typename MutexType>
inline void DebugAssertMutexHeld(MutexType* mutex)
    TENSORSTORE_THREAD_ANNOTATION_ATTRIBUTE(assert_capability(mutex)) {
  mutex->AssertHeld();
}
#else
template <typename MutexType>
inline void DebugAssertMutexHeld(MutexType* mutex)
    TENSORSTORE_THREAD_ANNOTATION_ATTRIBUTE(assert_capability(mutex)) {}
#endif

// ScopedUnlock temporarily unlocks a mutex which has been locked
// by std::unique_lock.
class ScopedUnlock {
 public:
  explicit ScopedUnlock(absl::Mutex& m) noexcept : mutex_(m) {
    DebugAssertMutexHeld(&mutex_);
    mutex_.unlock();
  }
  ~ScopedUnlock() noexcept { mutex_.lock(); }

 private:
  absl::Mutex& mutex_;
};

}  // namespace internal
}  // namespace tensorstore

#undef TENSORSTORE_THREAD_ANNOTATION_ATTRIBUTE

#endif  //  TENSORSTORE_INTERNAL_MUTEX_H_
