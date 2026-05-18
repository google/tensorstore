// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/os/file_descriptor.h"

#include <cerrno>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/file_test_hooks.h"
#include "tensorstore/internal/testing/test_hook.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal_tracing::LoggedTraceSpan;

namespace tensorstore {
namespace internal_os {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");

}  // namespace

absl::Status CloseFileDescriptor(FileDescriptor fd) {
  TENSORSTORE_INVOKE_TEST_HOOK(CloseOpTag, fd);
#ifdef _WIN32
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (!::CloseHandle(fd)) {
    auto error = ::GetLastError();
    tspan.Log("::GetLastError()", error);
    return tensorstore::internal::StatusFromOsError(error).Format(
        "Failed to close handle");
  }
  return absl::OkStatus();
#else
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  while (true) {
    if (::close(fd) == 0) {
      return absl::OkStatus();
    }
    const auto error = errno;

#if !defined(__linux__)
    // On non-linux posix systems, EINTR may be returned from close.
    if (error == EINTR) continue;
#endif
    tspan.Log("errno", error);
    return tensorstore::internal::StatusFromOsError(error).Format(
        "Failed to close file descriptor");
  }
#endif
}

}  // namespace internal_os
}  // namespace tensorstore
