
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

#if !defined(_WIN32)
#error "Use file_info_posix.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "tensorstore/internal/os/file_info.h"
// Maintain include ordering here:

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/file_descriptor.h"
#include "tensorstore/internal/os/wstring.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

using ::tensorstore::internal::ConvertUTF8ToWindowsWide;
using ::tensorstore::internal::ConvertWindowsWideToUTF8;
using ::tensorstore::internal::StatusFromOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

namespace tensorstore {
namespace internal_os {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("file_detail");

template <typename T>
inline absl::Time FileTimeToAbslTime(const T& file_time) {
  // Windows FILETIME is the number of 100-nanosecond intervals since the
  // Windows epoch (1601-01-01) which is 11644473600 seconds before the unix
  // epoch (1970-01-01).
  uint64_t windowsTicks =
      (static_cast<uint64_t>(file_time.dwHighDateTime) << 32) |
      static_cast<uint64_t>(file_time.dwLowDateTime);

  return absl::UnixEpoch() +
         absl::Seconds((windowsTicks / 10000000) - 11644473600ULL) +
         absl::Nanoseconds(windowsTicks % 10000000);
}

}  // namespace

absl::Time FileInfo::GetMTime() const {
  return FileTimeToAbslTime(impl.ftLastWriteTime);
}

absl::Time FileInfo::GetCTime() const {
  return FileTimeToAbslTime(impl.ftCreationTime);
}

uint32_t FileInfo::GetMode() const {
  if (impl.dwFileAttributes & FILE_ATTRIBUTE_READONLY) {
    return 0444;  // read-only
  } else {
    return 0666;  // read/write
  }
}

absl::Status GetFileInfo(HANDLE fd, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"handle", fd}});

  if (::GetFileInformationByHandle(fd, &info->impl)) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(::GetLastError());
  return std::move(tspan).EndWithStatus(std::move(status));
}

absl::Status GetFileInfo(const std::string& path, FileInfo* info) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"path", path}});

  // The typedef uses BY_HANDLE_FILE_INFO, which includes device and index
  // metadata, and requires an open handle.
  std::wstring wpath;
  TENSORSTORE_RETURN_IF_ERROR(ConvertUTF8ToWindowsWide(path, wpath));
  UniqueFileDescriptor stat_fd(::CreateFileW(
      wpath.c_str(), /*dwDesiredAccess=*/GENERIC_READ,
      /*dwShareMode=*/FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
      /*lpSecurityAttributes=*/nullptr,
      /*dwCreationDisposition=*/OPEN_EXISTING,
      /*dwFlagsAndAttributes=*/FILE_FLAG_BACKUP_SEMANTICS,
      /*hTemplateFile=*/nullptr));

  if (stat_fd.valid() &&
      ::GetFileInformationByHandle(stat_fd.get(), &info->impl)) {
    return absl::OkStatus();
  }
  auto status = StatusFromOsError(::GetLastError(),
                                  "Failed to stat file: ", QuoteString(path));
  return std::move(tspan).EndWithStatus(std::move(status));
}

}  // namespace internal_os
}  // namespace tensorstore
