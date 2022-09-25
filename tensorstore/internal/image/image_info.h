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

#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_INFO_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_INFO_H_

#include <stddef.h>

#include <iosfwd>

#include "tensorstore/data_type.h"

namespace tensorstore {
namespace internal_image {

/// Describes the basic properties of an image.
/// NOTE: Other useful fields (such as channels, orientation, etc) aren't
/// available; however as additional image requirements develop the structure
/// may be augmented.
struct ImageInfo {
  int32_t height = 0;
  int32_t width = 0;
  int32_t num_components = 0;
  DataType dtype = dtype_v<uint8_t>;

  friend bool operator==(const ImageInfo& a, const ImageInfo& b);
  friend bool operator!=(const ImageInfo& a, const ImageInfo& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const ImageInfo& info);
};

/// Returns the size of the buffer (in bytes) required to decode the image
/// described by ImageInfo.
size_t ImageRequiredBytes(const ImageInfo& a);

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_IMAGE_INFO_H_
