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

#include "tensorstore/internal/os/hugepages.h"

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "absl/flags/flag.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/util/result.h"

// Attempt to use hugepages for allocations.
ABSL_FLAG(std::optional<size_t>, tensorstore_hugepage_threshold, std::nullopt,
          "Threshold to allocate hugepages via mmap. When unset os-default "
          "allocation is used. Overrides TENSORSTORE_HUGEPAGE_THRESHOLD.");

namespace tensorstore {
namespace internal_os {
namespace {

constexpr size_t kHugepageThreshold = internal_os::kHugepageSize * 0.8;

size_t GetHugepageThreshold() {
  auto v = internal::GetFlagOrEnvValue(FLAGS_tensorstore_hugepage_threshold,
                                       "TENSORSTORE_HUGEPAGE_THRESHOLD");
  return v.has_value() ? std::max(*v, kHugepageThreshold)
                       : std::numeric_limits<size_t>::max();
}

}  // namespace

MemoryRegion AllocateHugePageRegionWithFallback(size_t alignment, size_t size) {
  if (auto hugepage_mode = internal_os::UseAllocateHugePageRegion();
      hugepage_mode.ok() && *hugepage_mode && size >= GetHugepageThreshold()) {
    // Allocate hugepages from mmap when manual hugepage allocation is required
    // and when the requested allocation is larger than the threshold.
    size_t hugepage_alignment = std::max(alignment, internal_os::kHugepageSize);
    auto huge_page_region =
        internal_os::AllocateHugePageRegion(hugepage_alignment, size);
    if (huge_page_region.ok()) {
      return *std::move(huge_page_region);
    }
  }
  if (alignment > 1) {
    return internal_os::AllocateAlignedRegion(alignment, size);
  }
  return internal_os::AllocateHeapRegion(size);
}

}  // namespace internal_os
}  // namespace tensorstore
