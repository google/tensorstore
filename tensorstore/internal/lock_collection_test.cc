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

#include <mutex>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/concurrent_testutil.h"

namespace {
using tensorstore::internal::LockCollection;

TEST(LockCollectionTest, Empty) {
  LockCollection c;
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    // Nothing to check.
  }
}

TEST(LockCollectionTest, SingleShared) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterShared(m);
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
}

TEST(LockCollectionTest, SingleSharedDuplicate) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterShared(m);
  c.RegisterShared(m);
  c.RegisterShared(m);
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
}

TEST(LockCollectionTest, SingleExclusive) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterExclusive(m);
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertHeld();
  }
  m.AssertNotHeld();
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertHeld();
  }
  m.AssertNotHeld();
}

TEST(LockCollectionTest, SingleExclusiveDuplicate) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterShared(m);
  c.RegisterExclusive(m);
  c.RegisterShared(m);
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m.AssertHeld();
  }
  m.AssertNotHeld();
}

TEST(LockCollectionTest, Multiple) {
  absl::Mutex m[3];
  LockCollection c;
  c.RegisterShared(m[0]);
  c.RegisterExclusive(m[0]);
  c.RegisterShared(m[1]);
  c.RegisterShared(m[0]);
  c.RegisterShared(m[2]);
  c.RegisterShared(m[1]);
  c.RegisterShared(m[1]);
  c.RegisterShared(m[2]);
  {
    std::unique_lock<LockCollection> guard(c, std::try_to_lock);
    ASSERT_TRUE(guard);
    m[0].AssertHeld();
    m[1].AssertReaderHeld();
    m[2].AssertReaderHeld();
  }
  m[0].AssertNotHeld();
  m[1].AssertNotHeld();
  m[2].AssertNotHeld();
}

// TestConcurrent can cause extreme system slowdowns on MS Windows, resulting in
// test timeouts.  These tests aren't critical and don't have any
// platform-specific behavior that shouldn't already be tested by Abseil, so we
// skip them on Windows.
#if !defined(_WIN32)

// Tests that LockCollection avoids deadlock.
TEST(LockCollectionTest, MultipleConcurrentExclusive) {
  constexpr static size_t kNumMutexes = 3;
  absl::Mutex m[kNumMutexes];
  constexpr static size_t kNumCollections = 3;
  LockCollection c[kNumCollections];

  // c[0] and c[1] get the first two permutations of the mutex order.
  // c[2] gets every other permutation.

  std::array<int, kNumMutexes> mutex_indices;
  absl::c_iota(mutex_indices, 0);

  const auto RegisterFromPermutation = [&](LockCollection& lock_collection) {
    for (auto i : mutex_indices) lock_collection.RegisterExclusive(m[i]);
  };
  RegisterFromPermutation(c[0]);
  absl::c_next_permutation(mutex_indices);
  RegisterFromPermutation(c[1]);
  while (absl::c_next_permutation(mutex_indices)) {
    c[2] = LockCollection();
    RegisterFromPermutation(c[2]);
    tensorstore::internal::TestConcurrent<kNumCollections>(
        /*num_iterations=*/100,
        /*initialize=*/[] {},
        /*finalize=*/[] {},
        /*concurrent_op=*/
        [&](size_t i) {
          std::unique_lock<LockCollection> guard(c[i], std::try_to_lock);
          ASSERT_TRUE(guard);
        });
  }
}

// Tests that LockCollection avoids deadlock when combining shared and
// exclusive locks.
TEST(LockCollectionTest, MultipleConcurrentExclusiveShared) {
  constexpr static size_t kNumMutexes = 3;
  absl::Mutex m[kNumMutexes];
  constexpr static size_t kNumCollections = 3;
  constexpr static size_t kNumSharedCombinations = size_t(1) << kNumMutexes;
  LockCollection c[kNumCollections];

  // c[0] and c[1] get the first two permutations of the mutex order.
  // c[2] gets every other permutation.

  std::array<int, kNumMutexes> mutex_indices;
  absl::c_iota(mutex_indices, 0);

  const auto RegisterFromPermutation = [&](LockCollection& lock_collection,
                                           size_t shared_bit_vector) {
    for (auto i : mutex_indices) {
      if ((shared_bit_vector >> i) & i) {
        lock_collection.RegisterShared(m[i]);
      } else {
        lock_collection.RegisterExclusive(m[i]);
      }
    }
  };
  RegisterFromPermutation(c[0], 0);
  absl::c_next_permutation(mutex_indices);
  RegisterFromPermutation(c[1], ~size_t(0));
  while (absl::c_next_permutation(mutex_indices)) {
    for (size_t shared_bit_vector = 0;
         shared_bit_vector < kNumSharedCombinations; ++shared_bit_vector) {
      c[2] = LockCollection();
      RegisterFromPermutation(c[2], shared_bit_vector);
      tensorstore::internal::TestConcurrent<kNumCollections>(
          /*num_iterations=*/20,
          /*initialize=*/[] {},
          /*finalize=*/[] {},
          /*concurrent_op=*/
          [&](size_t i) {
            std::unique_lock<LockCollection> guard(c[i], std::try_to_lock);
            EXPECT_TRUE(guard);
          });
    }
  }
}

#endif  // !defined(_WIN32)

struct LoggingLockable;
using LockLog = std::vector<std::pair<LoggingLockable*, bool>>;
struct LoggingLockable {
  LockLog& log;
  bool fail;
};

/// Tests handling of a lock failure.
TEST(LockCollectionTest, Fail) {
  LockLog log;
  LoggingLockable lockables[4] = {
      LoggingLockable{log, false},
      LoggingLockable{log, false},
      LoggingLockable{log, true},
      LoggingLockable{log, true},
  };
  constexpr auto lock_function = [](void* data, bool lock) -> bool {
    auto* lockable = static_cast<LoggingLockable*>(data);
    lockable->log.emplace_back(lockable, lock);
    if (lock && lockable->fail) return false;
    return true;
  };
  LockCollection c;
  for (auto& lockable : lockables) {
    c.Register(&lockable, lock_function, false);
  }
  std::unique_lock<LockCollection> guard(c, std::try_to_lock);
  EXPECT_FALSE(guard);
  EXPECT_THAT(log,
              ::testing::ElementsAre(::testing::Pair(&lockables[0], true),
                                     ::testing::Pair(&lockables[1], true),
                                     ::testing::Pair(&lockables[2], true),
                                     ::testing::Pair(&lockables[1], false),
                                     ::testing::Pair(&lockables[0], false)));
}

}  // namespace
