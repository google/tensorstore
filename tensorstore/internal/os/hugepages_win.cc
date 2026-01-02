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

#ifndef _WIN32
#error "Use hugepages.cc instead."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "tensorstore/internal/os/hugepages.h"
//

#include <stddef.h>
#include <windows.h>

#include <cassert>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {
namespace {

using ::tensorstore::internal::StatusWithOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("hugepages_detail");

// Helper to enable SeLockMemoryPrivilege.
// Returns 0 on success, or Windows error code on failure.
// See: https://devblogs.microsoft.com/oldnewthing/20110128-00/?p=11643
DWORD EnableLockMemoryPrivilege() {
  HANDLE hToken;
  if (!OpenProcessToken(::GetCurrentProcess(),
                        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
    return ::GetLastError();
  }

  TOKEN_PRIVILEGES tp;
  tp.PrivilegeCount = 1;
  tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

  // LookupPrivilegeValue returns the LUID of the SE_LOCK_MEMORY privilege.
  if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME,
                            &tp.Privileges[0].Luid)) {
    DWORD error = ::GetLastError();
    ::CloseHandle(hToken);
    return error;
  }

  auto success = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL);
  DWORD error = ::GetLastError();
  ::CloseHandle(hToken);
  if (success && error == ERROR_SUCCESS) {
    return 0;
  }
  return error;
}

struct LargePageInfo {
  size_t page_size = 0;
  absl::Status status = absl::OkStatus();
};

const LargePageInfo& GetLargePageInfo() {
  static const LargePageInfo info = []() -> LargePageInfo {
    size_t page_size = ::GetLargePageMinimum();
    if (page_size == 0) {
      return {0, absl::UnimplementedError(
                     "Large pages not supported by hardware or OS.")};
    }
    DWORD error = EnableLockMemoryPrivilege();
    if (error != 0) {
      return {0, StatusWithOsError(
                     absl::StatusCode::kPermissionDenied, error,
                     "Failed to enable SeLockMemoryPrivilege (required for "
                     "large pages)")};
    }
    return {page_size, absl::OkStatus()};
  }();
  return info;
}

void FreeLargePageRegion(char* data, size_t size) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"size", size}});
  if (size == 0 || !data) {
    return;
  }
  ::VirtualFree(data, 0, MEM_RELEASE);
}

}  // namespace

Result<bool> UseAllocateHugePageRegion() {
  const auto& lp_info = GetLargePageInfo();
  if (lp_info.status.ok()) {
    return true;
  }
  return lp_info.status;
}

Result<MemoryRegion> AllocateHugePageRegion(size_t alignment, size_t size) {
  const auto& lp_info = GetLargePageInfo();
  TENSORSTORE_RETURN_IF_ERROR(lp_info.status);

  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"alignment", alignment}, {"size", size}});
  if (size == 0) {
    return MemoryRegion(nullptr, 0, FreeLargePageRegion);
  }
  assert((alignment & (alignment - 1)) == 0);
  // Maybe Return an error if alignment is less than page size.
  size_t aligned_size = RoundUpTo(size, lp_info.page_size);

  // alignment is ignored; VirtualAlloc with MEM_LARGE_PAGES allocates memory
  // aligned to page_size.
  void* ptr = ::VirtualAlloc(NULL, aligned_size,
                             MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                             PAGE_READWRITE);

  if (ptr == NULL) {
    auto status = StatusWithOsError(absl::StatusCode::kResourceExhausted,
                                    ::GetLastError(),
                                    "VirtualAlloc with MEM_LARGE_PAGES failed");
    return std::move(tspan).EndWithStatus(std::move(status));
  }
  return MemoryRegion(static_cast<char*>(ptr), aligned_size,
                      FreeLargePageRegion);
}

}  // namespace internal_os
}  // namespace tensorstore
