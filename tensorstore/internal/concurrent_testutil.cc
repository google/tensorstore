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

#ifdef _WIN32

#include "tensorstore/internal/concurrent_testutil.h"

#include <windows.h>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"

namespace tensorstore {
namespace internal {

TestConcurrentLock::TestConcurrentLock() {
  mutex_ = ::CreateMutexA(/*lpMutexAttributes=*/nullptr,
                          /*bInitialOwner=*/FALSE,
                          /*lpName=*/"TensorStoreTestConcurrentMutex");
  ABSL_CHECK(mutex_ != nullptr);
  if (::WaitForSingleObject(mutex_, 1 /*ms*/) != 0) {
    ABSL_LOG(INFO) << "Waiting on WIN32 Concurrent Lock";
    ::WaitForSingleObject(mutex_, INFINITE);
  }
}

TestConcurrentLock::~TestConcurrentLock() {
  ABSL_CHECK(::ReleaseMutex(mutex_));
  ::CloseHandle(mutex_);
}

void MaybeYield() { ::Sleep(0); }

}  // namespace internal
}  // namespace tensorstore
#endif
