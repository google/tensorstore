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
    std::lock_guard<LockCollection> guard{c};
    // Nothing to check.
  }
}

TEST(LockCollectionTest, SingleShared) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterShared(m);
  {
    std::lock_guard<LockCollection> guard{c};
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
  {
    std::lock_guard<LockCollection> guard{c};
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
    std::lock_guard<LockCollection> guard{c};
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
  {
    std::lock_guard<LockCollection> guard{c};
    m.AssertReaderHeld();
  }
  m.AssertNotHeld();
}

TEST(LockCollectionTest, SingleExclusive) {
  absl::Mutex m;
  LockCollection c;
  c.RegisterExclusive(m);
  {
    std::lock_guard<LockCollection> guard{c};
    m.AssertHeld();
  }
  m.AssertNotHeld();
  {
    std::lock_guard<LockCollection> guard{c};
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
    std::lock_guard<LockCollection> guard{c};
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
    std::lock_guard<LockCollection> guard{c};
    m[0].AssertHeld();
    m[1].AssertReaderHeld();
    m[2].AssertReaderHeld();
  }
  m[0].AssertNotHeld();
  m[1].AssertNotHeld();
  m[2].AssertNotHeld();
}

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
        [&](size_t i) { std::lock_guard<LockCollection> guard(c[i]); });
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
          [&](size_t i) { std::lock_guard<LockCollection> guard(c[i]); });
    }
  }
}

}  // namespace
