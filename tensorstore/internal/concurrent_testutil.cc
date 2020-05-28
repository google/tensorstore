#ifdef _WIN32

#include "tensorstore/internal/concurrent_testutil.h"

#include <windows.h>

#include "tensorstore/util/assert_macros.h"

namespace tensorstore {
namespace internal {

TestConcurrentLock::TestConcurrentLock() {
  mutex_ = ::CreateMutexA(/*lpMutexAttributes=*/nullptr,
                          /*bInitialOwner=*/FALSE,
                          /*lpName=*/"TensorStoreTestConcurrentMutex");
  TENSORSTORE_CHECK(mutex_ != nullptr);
  ::WaitForSingleObject(mutex_, INFINITE);
}

TestConcurrentLock::~TestConcurrentLock() {
  TENSORSTORE_CHECK(::ReleaseMutex(mutex_));
  ::CloseHandle(mutex_);
}

}  // namespace internal
}  // namespace tensorstore
#endif
