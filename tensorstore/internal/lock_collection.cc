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

#include "tensorstore/internal/lock_collection.h"

#include <stddef.h>

#include <algorithm>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace internal {

bool LockCollection::MutexSharedLockFunction(void* mutex, bool lock)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto& m = *static_cast<absl::Mutex*>(mutex);
  if (lock) {
    m.lock_shared();
  } else {
    m.unlock_shared();
  }
  return true;
}

bool LockCollection::MutexExclusiveLockFunction(void* mutex, bool lock)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto& m = *static_cast<absl::Mutex*>(mutex);
  if (lock) {
    m.lock();
  } else {
    m.unlock();
  }
  return true;
}

bool LockCollection::try_lock() {
  if (locks_.size() > 1) {
    // Sort by `tagged_pointer` value.  Since the tag bit is the least
    // significant bit, this groups together all entries with the same data
    // pointer, ensuring a consistent lock acquisition order based on the
    // address, and additionally ensures entries with a tag of `0` (meaning
    // exclusive) occur first in each group.
    std::sort(locks_.begin(), locks_.end(), [](const Entry& a, const Entry& b) {
      return a.tagged_pointer < b.tagged_pointer;
    });
    // Remove all but the first occurrence of each data pointer.  Because
    // exclusive lock requests are ordered first, this ensures an exclusive lock
    // takes precedence over a shared lock.
    locks_.erase(std::unique(locks_.begin(), locks_.end(),
                             [](const Entry& a, const Entry& b) {
                               return a.data() == b.data();
                             }),
                 locks_.end());
  }
  size_t i = 0, size = locks_.size();
  auto* locks = locks_.data();
  for (; i < size; ++i) {
    auto& entry = locks[i];
    if (!entry.lock_function(entry.data(), /*lock=*/true)) {
      while (i > 0) {
        --i;
        auto& prev_entry = locks[i];
        prev_entry.lock_function(prev_entry.data(), /*lock=*/false);
      }
      return false;
    }
  }
  return true;
}

void LockCollection::unlock() {
  for (const auto& entry : locks_) {
    entry.lock_function(entry.data(), /*lock=*/false);
  }
}

void LockCollection::clear() { locks_.clear(); }

}  // namespace internal
}  // namespace tensorstore
