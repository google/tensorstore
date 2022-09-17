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

#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_READER_H_

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

/// Base class for reading images from a riegeli::Reader stream.
class ImageReader {
 public:
  virtual ~ImageReader() = default;

  /// Initialize the codec to read an image from the provided reader.
  /// If Initialize returns successfully, then GetFrameCount and GetImageInfo
  /// may be called to retrieve information about the image.
  /// 'reader' must outlive the ImageReader instance.
  ///
  /// \checks `reader` is not null.
  virtual absl::Status Initialize(riegeli::Reader* reader) = 0;

  /// Returns the ImageInfo, which includes the width, height, etc., describing
  /// the next image returned by `Decode`.  This is only valid after
  /// Initialize() has returned success, otherwise the result is undefined.
  virtual ImageInfo GetImageInfo() = 0;

  /// Decodes the next available image into 'dest' using the provided options.
  /// Repeated calls will provide iteration through an animation or video, if
  /// available.
  ///
  /// The provided buffer must be at least the size returned by
  /// `ImageRequiredBytes`.
  ///
  /// \checks `Initialize` returned `absl::OkStatus()`
  virtual absl::Status Decode(tensorstore::span<unsigned char> dest) = 0;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_IMAGE_READER_H_
