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

#include <cstdlib>
#include <string_view>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"

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
  FlatCordBuilder() = default;
  explicit FlatCordBuilder(size_t size)
      : data_(static_cast<char*>(::malloc(size))), size_(size) {
    ABSL_CHECK(size == 0 || data_);
  }
  FlatCordBuilder(const FlatCordBuilder&) = delete;
  FlatCordBuilder& operator=(const FlatCordBuilder&) = delete;
  FlatCordBuilder(FlatCordBuilder&& other)
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  FlatCordBuilder& operator=(FlatCordBuilder&& other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }
  ~FlatCordBuilder() {
    if (data_) {
      ::free(data_);
    }
  }

  const char* data() const { return data_; }
  char* data() { return data_; }
  size_t size() const { return size_; }

  /// Resizes the buffer, preserving any existing data up to
  /// `min(size(), new_size)`.
  void resize(size_t new_size) {
    data_ = static_cast<char*>(::realloc(data_, new_size));
    size_ = new_size;
    ABSL_CHECK(new_size == 0 || data_);
  }

  absl::Cord Build() && {
    return absl::MakeCordFromExternal(release(), [](absl::string_view s) {
      ::free(const_cast<char*>(s.data()));
    });
  }

 private:
  /// Releases ownership of the buffer.  The caller must call `::free`.
  absl::string_view release() {
    absl::string_view view(data_, size_);
    data_ = nullptr;
    size_ = 0;
    return view;
  }

  char* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_FLAT_CORD_BUILDER_H_
