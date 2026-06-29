// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/os/shutdown.h"

#include "absl/log/absl_log.h"  // iwyu pragma: keep
#include "tensorstore/internal/source_location.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal {

void LogIfShutdownInProgress(SourceLocation loc) {
#if defined(_WIN32)
  typedef BOOLEAN(NTAPI * RtlDllShutdownInProgressFn)();
  RtlDllShutdownInProgressFn fn = []() -> RtlDllShutdownInProgressFn {
    const HMODULE ntdll = ::GetModuleHandleA("ntdll.dll");
    if (ntdll) {
      typedef BOOLEAN(NTAPI * RtlDllShutdownInProgressFn)();
      return reinterpret_cast<RtlDllShutdownInProgressFn>(
          GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
    }
    return nullptr;
  }();
  if (fn && fn() != FALSE) {
    ABSL_LOG_FIRST_N(ERROR, 1) << "Process is shutting down " << loc;
  }
#endif
}

}  // namespace internal
}  // namespace tensorstore
