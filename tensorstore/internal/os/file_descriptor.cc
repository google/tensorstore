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
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal_tracing::LoggedTraceSpan;

namespace tensorstore {
namespace internal_os {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");

}

/* static */
void FileDescriptorTraits::Close(FileDescriptor fd) {
#ifdef _WIN32
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (!::CloseHandle(fd)) {
    tspan.Log("::GetLastError()", ::GetLastError());
  }
#else
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  while (true) {
    if (::close(fd) == 0) {
      return;
    }
    if (errno == EINTR) continue;
    tspan.Log("errno", errno);
    return;
  }
#endif
}

}  // namespace internal_os
}  // namespace tensorstore
