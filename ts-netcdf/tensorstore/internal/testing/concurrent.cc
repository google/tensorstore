
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
#include "tensorstore/internal/testing/concurrent.h"  // IWYU pragma: keep

#include "absl/log/absl_check.h"  // IWYU pragma: keep
#include "absl/log/absl_log.h"    // IWYU pragma: keep

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace tensorstore {
namespace internal_testing {

#ifdef _WIN32

TestConcurrentLock::TestConcurrentLock() {
  handle_ = ::CreateMutexA(/*lpMutexAttributes=*/nullptr,
                           /*bInitialOwner=*/FALSE,
                           /*lpName=*/"TensorStoreTestConcurrentMutex");
  ABSL_CHECK(handle_ != nullptr);
  if (::WaitForSingleObject(handle_, 0 /*ms*/) != WAIT_OBJECT_0) {
    ABSL_LOG(INFO) << "Waiting on WIN32 Concurrent Lock";
    ABSL_CHECK(::WaitForSingleObject(handle_, INFINITE) == WAIT_OBJECT_0);
  }
}

TestConcurrentLock::~TestConcurrentLock() {
  ABSL_CHECK(::ReleaseMutex(handle_));
  ::CloseHandle(handle_);
}

#endif

}  // namespace internal_testing
}  // namespace tensorstore
