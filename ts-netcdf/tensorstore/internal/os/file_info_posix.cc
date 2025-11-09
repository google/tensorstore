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

#if defined(_WIN32)
#error "Use file_info_win.cc instead."
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "tensorstore/internal/os/file_info.h"
// Maintain include ordering here:

#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusFromOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

namespace tensorstore {
namespace internal_os {
namespace {
ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");
}

absl::Time FileInfo::GetMTime() const {
#if defined(__APPLE__)
  const struct ::timespec t = impl.st_mtimespec;
#else
  const struct ::timespec t = impl.st_mtim;
#endif
  return absl::FromTimeT(t.tv_sec) + absl::Nanoseconds(t.tv_nsec);
}

absl::Time FileInfo::GetCTime() const {
#if defined(__APPLE__)
  const struct ::timespec t = impl.st_ctimespec;
#else
  const struct ::timespec t = impl.st_ctim;
#endif
  return absl::FromTimeT(t.tv_sec) + absl::Nanoseconds(t.tv_nsec);
}

uint32_t FileInfo::GetMode() const { return impl.st_mode; }

absl::Status GetFileInfo(FileDescriptor fd, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"fd", fd}});

  PotentiallyBlockingRegion region;
  if (::fstat(fd, &info->impl) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno, "Failed to get file info");
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  PotentiallyBlockingRegion region;
  if (::stat(path.c_str(), &info->impl) == 0) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(errno);
  return std::move(tspan).EndWithStatus(std::move(status));
}

}  // namespace internal_os
}  // namespace tensorstore
