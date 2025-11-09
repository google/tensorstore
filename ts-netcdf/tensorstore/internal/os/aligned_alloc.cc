// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/os/aligned_alloc.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "absl/log/absl_log.h"
#include "tensorstore/internal/os/memory_region.h"

// Include system headers last to reduce impact of macros.
#include "tensorstore/internal/os/include_windows.h"

namespace tensorstore {
namespace internal_os {
namespace {

#ifdef _WIN32
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc?view=msvc-170
void AlignedFree(char* data, size_t size) { _aligned_free(data); }
#else
void AlignedFree(char* data, size_t size) { free(data); }
#endif

}  // namespace

MemoryRegion AllocatePageAlignedRegion(size_t alignment, size_t size) {
  if (size == 0) {
    return MemoryRegion(nullptr, 0, AlignedFree);
  }
  assert((alignment & (alignment - 1)) == 0);
  size_t rounded_size = 0;

#ifdef _WIN32
  void* p = _aligned_malloc(size, alignment);
#elif defined(__APPLE__)
  // https://reviews.llvm.org/D138196
  // ::aligned_alloc requires macos 15.0 or later. etc.
  void* p = nullptr;
  ::posix_memalign(&p, alignment, size);
#else
  rounded_size = (size + alignment - 1) & ~(alignment - 1);
  void* p = ::aligned_alloc(alignment, std::max(size, rounded_size));
#endif

  if (p == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate memory " << size;
  }
  return MemoryRegion(static_cast<char*>(p), std::max(size, rounded_size),
                      AlignedFree);
}

}  // namespace internal_os
}  // namespace tensorstore
