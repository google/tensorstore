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

#if !defined(_WIN32)
#if defined(__has_include)
#if __has_include(<pthread.h>)
#define TENSORSTORE_INTERNAL_USE_PTHREAD
#include <pthread.h>
#endif
#endif
#endif

#include <cstdio>
#include <cstdlib>

#include "absl/base/call_once.h"

namespace tensorstore {
namespace internal {
namespace {

[[noreturn]] void LogFatalOnFork() {
  std::fprintf(stderr,
               "aborting: fork() is not allowed since tensorstore uses "
               "internal threading\n");
  std::fflush(stderr);
  std::abort();
}

}  // namespace

void SetupLogFatalOnFork() {
#if defined(TENSORSTORE_INTERNAL_USE_PTHREAD)
  static absl::once_flag g_setup_pthread;
  absl::call_once(g_setup_pthread, &pthread_atfork, LogFatalOnFork, nullptr,
                  nullptr);
#endif
}

}  // namespace internal
}  // namespace tensorstore
