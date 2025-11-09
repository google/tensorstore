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

#ifndef TENSORSTORE_INTERNAL_OS_FORK_DETECTION_H_
#define TENSORSTORE_INTERNAL_OS_FORK_DETECTION_H_

#include <stdint.h>

#include <atomic>
namespace tensorstore {
namespace internal_os {

#if !defined(_WIN32)
extern std::atomic<bool> g_fork_detected;

/// Logs a message and aborts the process if a fork() call has been detected.
void AbortIfForkDetectedImpl();
#endif  // !defined(_WIN32)

/// Sets up fork detection.
///
/// Forks detection sets the g_fork_detected atomic to true via pthread_atfork.
void SetupForkDetection();

/// Aborts the process if a fork() call has been detected.
inline void AbortIfForkDetected() {
#if !defined(_WIN32)
  if (!g_fork_detected.load(std::memory_order_relaxed)) {
    return;
  }
  AbortIfForkDetectedImpl();
#endif  // !defined(_WIN32)
}

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_FORK_DETECTION_H_
