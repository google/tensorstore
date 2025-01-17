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

#ifndef TENSORSTORE_INTERNAL_OS_MEMORY_REGION_H_
#define TENSORSTORE_INTERNAL_OS_MEMORY_REGION_H_

#include <stddef.h>

#include <string_view>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

/// A region of memory that can be mapped into the process address space.
/// This may be backed by a file, anonymous memory, or the heap.
class MemoryRegion {
 public:
  ~MemoryRegion() {
    if (data_) {
      unmap_fn_(data_, size_);
    }
  }

  MemoryRegion(const MemoryRegion&) = delete;
  MemoryRegion& operator=(const MemoryRegion&) = delete;

  MemoryRegion(MemoryRegion&& other)
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        unmap_fn_(other.unmap_fn_) {}

  MemoryRegion& operator=(MemoryRegion&& other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(unmap_fn_, other.unmap_fn_);
    return *this;
  }

  size_t size() const { return size_; }
  const char* data() const { return data_; }
  char* data() { return data_; }

  std::string_view as_string_view() const {
    return std::string_view(data_, size_);
  }

  absl::Cord as_cord() &&;

 private:
  using unmap_fn = void (*)(char*, size_t);

  MemoryRegion(char* data, size_t size, unmap_fn unmap_fn)
      : data_(data), size_(size), unmap_fn_(unmap_fn) {
    ABSL_CHECK(size == 0 || data_);
  }

  friend Result<MemoryRegion> MemmapFileReadOnly(void*, size_t, size_t);
  friend Result<MemoryRegion> MemmapFileReadOnly(int, size_t, size_t);
  friend MemoryRegion AllocateHeapRegion(size_t);

  char* data_;
  size_t size_;
  unmap_fn unmap_fn_;
};

/// Try to allocate a region of memory backed the heap.
MemoryRegion AllocateHeapRegion(size_t size);

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_MEMORY_REGION_H_
