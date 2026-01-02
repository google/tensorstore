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

#ifndef __linux__
#error "Use hugepages.cc instead."
#endif

#include "tensorstore/internal/os/hugepages.h"
//

#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <ctime>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/internal/os/potentially_blocking_region.h"
#include "tensorstore/internal/tracing/logged_trace_span.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_os {
namespace {

using ::tensorstore::internal::PotentiallyBlockingRegion;
using ::tensorstore::internal::StatusWithOsError;
using ::tensorstore::internal_tracing::LoggedTraceSpan;

ABSL_CONST_INIT internal_log::VerboseFlag detail_logging("hugepages_detail");

void FreeHugePageRegion(char* data, size_t size) {
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1), {{"size", size}});
  if (size == 0 || !data) {
    return;
  }
  // Consider calling ::madvise(data, size, MADV_DONTNEED);

  int error;
  {
    internal::PotentiallyBlockingRegion region;
    error = ::munmap(data, size);
  }
  if (error == -1) {
    tspan.Log("errno", errno);
  }
}

Result<HugePageMode> GetShmemTransparentHugePageMode() {
  static auto result = []() -> Result<HugePageMode> {
    auto hugepages = internal_os::ReadAllToString(
        "/sys/kernel/mm/transparent_hugepage/shmem_enabled");
    if (!hugepages.ok()) {
      return hugepages.status();
    }
    if (absl::StrContains(hugepages.value(), "[always]")) {
      return HugePageMode::kAlways;
    } else if (absl::StrContains(hugepages.value(), "[madvise]")) {
      return HugePageMode::kMadvise;
    } else if (absl::StrContains(hugepages.value(), "[force]")) {
      return HugePageMode::kAlways;
    } else {
      // Includes [deny], [never] and unknown values.
      return HugePageMode::kNever;
    }
  }();
  return result;
}

// Reads Hugepagesize: from /proc/meminfo
size_t GetHugepageSize() {
  static size_t result = []() -> size_t {
    auto meminfo = internal_os::ReadAllToString("/proc/meminfo");
    if (!meminfo.ok()) {
      return 0;
    }
    for (const auto& line : absl::StrSplit(meminfo.value(), '\n')) {
      if (absl::StartsWith(line, "Hugepagesize:")) {
        absl::string_view line_view = line;
        line_view.remove_prefix(line.find(':') + 1);
        line_view = absl::StripAsciiWhitespace(line_view);
        size_t size;
        if (absl::SimpleAtoi(line_view, &size)) {
          return size * 1024;
        }
      }
    }
    return 0;
  }();
  return result;
}

}  // namespace

Result<HugePageMode> GetTransparentHugePageMode() {
  static auto result = []() -> Result<HugePageMode> {
    auto hugepages = internal_os::ReadAllToString(
        "/sys/kernel/mm/transparent_hugepage/enabled");
    if (!hugepages.ok()) {
      return hugepages.status();
    }
    if (absl::StrContains(hugepages.value(), "[always]")) {
      return HugePageMode::kAlways;
    } else if (absl::StrContains(hugepages.value(), "[madvise]")) {
      return HugePageMode::kMadvise;
    } else if (absl::StrContains(hugepages.value(), "[never]")) {
      return HugePageMode::kMadvise;
    } else {
      // Includes [never] and unknown values.
      return HugePageMode::kNever;
    }
  }();
  return result;
}

Result<bool> UseAllocateHugePageRegion() {
  static Result<bool> result = []() -> Result<bool> {
    if (auto hugepage_mode = GetTransparentHugePageMode();
        !hugepage_mode.ok()) {
      return hugepage_mode.status();
    } else if (*hugepage_mode == HugePageMode::kNever) {
      // Transparent huge pages are not supported, and hugetlbfs or similar are
      // required, however tensorstore doesn't currently support that.
      return false;
    } else if (*hugepage_mode == HugePageMode::kAlways) {
      // Huge pages are always enabled, no special allocation needed.
      return false;
    }

    if (auto shm_enabled = GetShmemTransparentHugePageMode();
        !shm_enabled.ok()) {
      return shm_enabled.status();
    } else if (*shm_enabled == HugePageMode::kNever) {
      return false;
    } else {
      return true;
    }
  }();
  return result;
}

Result<MemoryRegion> AllocateHugePageRegion(size_t alignment, size_t size) {
  if (auto shm_enabled = GetShmemTransparentHugePageMode(); !shm_enabled.ok()) {
    return shm_enabled.status();
  } else if (*shm_enabled == HugePageMode::kNever) {
    return absl::ResourceExhaustedError("hugepages allocation not allowed");
  }
  LoggedTraceSpan tspan(__func__, detail_logging.Level(1),
                        {{"alignment", alignment}, {"size", size}});
  if (size == 0) {
    return MemoryRegion(nullptr, 0, FreeHugePageRegion);
  }
  assert((alignment & (alignment - 1)) == 0);
  size = RoundUpTo(size, std::max(GetHugepageSize(), alignment));

  void* addr = [size]() -> void* {
    int flags =
        MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE | MAP_LOCKED | MAP_HUGETLB;
#if defined(MAP_HUGE_2MB)
    flags |= MAP_HUGE_2MB;
#endif
    internal::PotentiallyBlockingRegion region;
    return ::mmap(nullptr, size, PROT_READ | PROT_WRITE, flags,
                  /*fd*/ -1, 0);
  }();
  if (addr == MAP_FAILED || addr == nullptr) {
    auto status = StatusWithOsError(absl::StatusCode::kResourceExhausted, errno,
                                    "Failed to mmap huge page region");
    return std::move(tspan).EndWithStatus(std::move(status));
  }

  MemoryRegion region(static_cast<char*>(addr), size, FreeHugePageRegion);

  if (auto shm_mode = GetShmemTransparentHugePageMode();
      shm_mode.ok() && *shm_mode == HugePageMode::kMadvise) {
    // madvise the region to be huge pages.
    int error;
    int attempts = 0;
    // Retry madvise() when it returns EAGAIN.
    {
      internal::PotentiallyBlockingRegion region;
      do {
        error = ::madvise(addr, size, MADV_HUGEPAGE);
      } while (error == -1 && errno == EAGAIN && ++attempts < 3);
    }
    if (error == -1) {
      tspan.Log("errno", errno);
    }
  }

  return region;
}

}  // namespace internal_os
}  // namespace tensorstore
