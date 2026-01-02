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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif  // _WIN32

#include "tensorstore/internal/os/memory_region.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/division.h"

// Include system headers last to reduce impact of macros.
#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

namespace tensorstore {
namespace internal_os {
namespace {

void HeapFree(char* data, size_t size) { ::free(data); }

#ifdef _WIN32
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc?view=msvc-170
void AlignedFree(char* data, size_t size) { _aligned_free(data); }
#else
#define AlignedFree HeapFree
#endif

}  // namespace

absl::Cord MemoryRegion::as_cord() && {
  std::string_view string_view = as_string_view();
  data_ = nullptr;
  size_ = 0;
  return absl::MakeCordFromExternal(
      string_view, [unmap_fn = unmap_fn_](auto s) {
        MemoryRegion releaser(const_cast<char*>(s.data()), s.size(), unmap_fn);
      });
}

MemoryRegion AllocateHeapRegion(size_t size) {
  if (size == 0) {
    return MemoryRegion(nullptr, 0, HeapFree);
  }
  void* p = ::malloc(size);
  if (p == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate memory " << size;
  }
  return MemoryRegion(static_cast<char*>(p), size, HeapFree);
}

MemoryRegion AllocateAlignedRegion(size_t alignment, size_t size) {
  if (size == 0) {
    return MemoryRegion(nullptr, 0, AlignedFree);
  }
  assert((alignment & (alignment - 1)) == 0);
  size = RoundUpTo(size, alignment);
#ifdef _WIN32
  void* p = _aligned_malloc(size, alignment);
#elif defined(__APPLE__)
  // https://reviews.llvm.org/D138196
  // ::aligned_alloc requires macos 15.0 or later. etc.
  void* p = nullptr;
  ::posix_memalign(&p, alignment, size);
#else
  // Round the size up to the next multiple of the alignment.
  void* p = ::aligned_alloc(alignment, size);
#endif

  if (p == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate memory " << size;
  }
  return MemoryRegion(static_cast<char*>(p), size, AlignedFree);
}

}  // namespace internal_os
}  // namespace tensorstore
