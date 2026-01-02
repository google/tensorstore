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

#ifndef TENSORSTORE_INTERNAL_OS_HUGEPAGES_H_
#define TENSORSTORE_INTERNAL_OS_HUGEPAGES_H_

#include <stddef.h>

#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

constexpr size_t kHugepageSize = 2 * 1024 * 1024;  // 2 MiB

#if defined(__linux__)
enum class HugePageMode {
  kNever = 0,  // transparent huge pages are disabled.
  kAlways,     // transparent huge pages are always enabled, no action needed.
  kMadvise,    // transparent huge pages are allowed with madvise.
};

/// Returns the transparent huge page mode, which indicates whether transparent
/// huge pages are enabled. When this is `kAlways`, no special action needs to
/// be taken to use huge pages since tcmalloc handles the allocation.
Result<HugePageMode> GetTransparentHugePageMode();
#endif  // __linux__

/// Returns true if AllocateHugePageRegion() should be used for hugepage memory
/// allocations.
///
/// On Linux, AllocateHugePageRegion() only supports transparent huge pages, not
/// hugetlbfs, and should be used when:
/// 1. /sys/kernel/mm/transparent_hugepage/enabled is set to [madvise]
/// 2. /sys/kernel/mm/transparent_hugepage/shmem_enabled is not set to [never]
///
/// See https://docs.kernel.org/admin-guide/mm/transhuge.html
///
/// On Windows, UseAllocateHugePageRegion() indicates whether allocating memory
/// using VirtualAlloc with MEM_LARGE_PAGES is supported by attempting to use
/// the SE_LOCK_MEMORY privilege. This privilege must be enabled on the process.
///
/// See: https://devblogs.microsoft.com/oldnewthing/20110128-00/?p=11643
///
/// On other platforms, this always returns false.
///
Result<bool> UseAllocateHugePageRegion();

/// Allocates a region of memory that is backed by huge pages.
Result<MemoryRegion> AllocateHugePageRegion(size_t alignment, size_t size);

/// Allocates a region of memory that is backed by huge pages if possible,
/// otherwise falls back to AllocateAlignedRegion().
MemoryRegion AllocateHugePageRegionWithFallback(size_t alignment, size_t size);

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_HUGEPAGES_H_
