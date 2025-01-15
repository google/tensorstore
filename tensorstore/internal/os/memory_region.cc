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

#include "tensorstore/internal/os/memory_region.h"

#include <stddef.h>
#include <stdio.h>

#include <cstdlib>
#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal_os {
namespace {

void FreeHeap(char* data, size_t size) { ::free(data); }

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
    return MemoryRegion(nullptr, 0, FreeHeap);
  }
  void* p = ::malloc(size);
  if (p == nullptr) {
    ABSL_LOG(FATAL) << "Failed to allocate memory " << size;
  }
  return MemoryRegion(static_cast<char*>(p), size, FreeHeap);
}

}  // namespace internal_os
}  // namespace tensorstore
