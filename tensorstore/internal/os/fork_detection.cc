// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/os/fork_detection.h"

#if !defined(_WIN32)

#include <pthread.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "absl/base/call_once.h"

namespace tensorstore {
namespace internal_os {

// Use a single atomic bool for fork detection.  If this is not sufficient,
// another strategy is to allocate one page, initialize it to non-zero, then use
// MADV_WIPEONFORK/MADV_INHERIT_ZERO. to zero the page on fork.
std::atomic<bool> g_fork_detected{false};

namespace {

absl::once_flag g_once;

void PthreadZeroOnFork() { g_fork_detected.store(true); }

}  // namespace

void SetupForkDetection() {
  // InitializeForkDetection sets up the fork detection memory region.
  absl::call_once(g_once, &pthread_atfork, nullptr, nullptr, PthreadZeroOnFork);
}

void AbortIfForkDetectedImpl() {
  std::fprintf(stderr,
               "aborting: fork() use detected, which is not allowed due to "
               "internal threading use by tensorstore\n");
  std::fflush(stderr);
  std::abort();
}

}  // namespace internal_os
}  // namespace tensorstore

#else
namespace tensorstore {
namespace internal_os {

void SetupForkDetection() {}

}  // namespace internal_os
}  // namespace tensorstore

#endif  // !defined(_WIN32)
