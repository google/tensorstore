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

#ifndef TENSORSTORE_INTERNAL_FLAT_CORD_BUILDER_H_
#define TENSORSTORE_INTERNAL_FLAT_CORD_BUILDER_H_

#include <stddef.h>

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/os/memory_region.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Builds a Cord guaranteed to be flat.
///
/// Normally, a Cord is represented by a tree of chunks of a fixed maximum size.
/// This results in a Cord containing just a single chunk, which is more
/// efficient if the full size is known in advance and the entire string is
/// likely to be retained.
class FlatCordBuilder {
 public:
  explicit FlatCordBuilder(size_t size) : FlatCordBuilder(size, size) {}

  explicit FlatCordBuilder(internal_os::MemoryRegion region)
      : region_(std::move(region)), inuse_(region_.size()) {}

  FlatCordBuilder(size_t size, size_t inuse)
      : FlatCordBuilder(internal_os::AllocateHeapRegion(size), inuse) {}

  FlatCordBuilder(internal_os::MemoryRegion region, size_t inuse)
      : region_(std::move(region)), inuse_(inuse) {
    ABSL_CHECK(inuse <= region_.size());
  }

  FlatCordBuilder(const FlatCordBuilder&) = delete;
  FlatCordBuilder& operator=(const FlatCordBuilder&) = delete;
  FlatCordBuilder(FlatCordBuilder&& other)
      : region_(std::move(other.region_)),
        inuse_(std::exchange(other.inuse_, 0)) {}

  FlatCordBuilder& operator=(FlatCordBuilder&& other) {
    region_ = std::move(other.region_);
    inuse_ = std::exchange(other.inuse_, 0);
    return *this;
  }

  ~FlatCordBuilder() = default;

  const char* data() const { return region_.data(); }
  char* data() { return region_.data(); }
  size_t size() const { return region_.size(); }
  size_t available() const { return region_.size() - inuse_; }

  void set_inuse(size_t size) {
    ABSL_CHECK(size <= region_.size());
    inuse_ = size;
  }

  tensorstore::span<char> available_span() {
    return tensorstore::span(region_.data() + inuse_, available());
  }

  /// Append data to the builder.
  void Append(std::string_view sv) {
    if (ABSL_PREDICT_FALSE(sv.empty())) return;
    ABSL_CHECK(sv.size() <= available());
    ::memcpy(region_.data() + inuse_, sv.data(), sv.size());
    inuse_ += sv.size();
  }

  absl::Cord Build() && {
    if (inuse_ == region_.size()) {
      return std::move(region_).as_cord();
    }
    return std::move(region_).as_cord().Subcord(0, inuse_);
  }

 private:
  internal_os::MemoryRegion region_;
  size_t inuse_ = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_FLAT_CORD_BUILDER_H_
