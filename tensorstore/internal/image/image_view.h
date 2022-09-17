// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_VIEW_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_VIEW_H_

#include <assert.h>

#include <cstddef>

#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

/// A simple row-oriented view into an image buffer.
/// Used as an implementation detail to aid in strided buffer access.
class ImageView {
 public:
  using index_type = std::ptrdiff_t;

  /// Construct an image which references the data (unowned);
  ImageView(const ImageInfo& info, tensorstore::span<unsigned char> data);

  constexpr index_type row_stride() const noexcept { return row_stride_; }
  constexpr index_type row_stride_bytes() const noexcept {
    return row_stride_ * dtype_.size();
  }

  constexpr DataType dtype() const noexcept { return dtype_; }

  /// Returns a pointer to the first element.
  constexpr tensorstore::span<unsigned char> data() const noexcept {
    return data_;
  }

  constexpr tensorstore::span<unsigned char> data_row(
      size_t row, size_t col = 0) const noexcept {
    assert(row_stride_ != 0);
    assert(col < row_stride_bytes());
    return {data_.data() + row * row_stride_bytes() + col,
            row_stride_bytes() - static_cast<index_type>(col)};
  }

 private:
  tensorstore::span<unsigned char> data_;
  DataType dtype_;
  index_type row_stride_ = 0;
};

inline ImageView MakeWriteImageView(
    const ImageInfo& info, tensorstore::span<const unsigned char> source) {
  return ImageView(
      info, tensorstore::span<unsigned char>(
                const_cast<unsigned char*>(source.data()), source.size()));
}

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_IMAGE_VIEW_H_
