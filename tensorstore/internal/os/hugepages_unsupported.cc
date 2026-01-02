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

#if defined(_WIN32)
#error "Use hugepages_win.cc instead."
#elif defined(__linux__)
#error "Use hugepages_linux.cc instead."
#endif

#include "tensorstore/internal/os/hugepages.h"
//

#include <stddef.h>

#include "absl/status/status.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

Result<bool> UseAllocateHugePageRegion() { return false; }

Result<MemoryRegion> AllocateHugePageRegion(size_t alignment, size_t size) {
  return absl::UnimplementedError(
      "Transparent huge pages are not implemented on this platform");
}

}  // namespace internal_os
}  // namespace tensorstore
