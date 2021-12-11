#ifdef _WIN32

#include "tensorstore/internal/concurrent_testutil.h"

#include <windows.h>

#include "tensorstore/internal/logging.h"
#include "tensorstore/util/assert_macros.h"

namespace tensorstore {
namespace internal {

TestConcurrentLock::TestConcurrentLock() {
  mutex_ = ::CreateMutexA(/*lpMutexAttributes=*/nullptr,
                          /*bInitialOwner=*/FALSE,
                          /*lpName=*/"TensorStoreTestConcurrentMutex");
  TENSORSTORE_CHECK(mutex_ != nullptr);
  if (::WaitForSingleObject(mutex_, 1 /*ms*/) != 0) {
    TENSORSTORE_LOG("Waiting on WIN32 Concurrent Lock");
    ::WaitForSingleObject(mutex_, INFINITE);
  }
}

TestConcurrentLock::~TestConcurrentLock() {
  TENSORSTORE_CHECK(::ReleaseMutex(mutex_));
  ::CloseHandle(mutex_);
}

void MaybeYield() { ::Sleep(0); }

}  // namespace internal
}  // namespace tensorstore
#endif
