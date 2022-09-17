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

#ifndef TENSORSTORE_INTERNAL_IMAGE_IMAGE_WRITER_H_
#define TENSORSTORE_INTERNAL_IMAGE_IMAGE_WRITER_H_

#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

/// Base class for encoding images.
class ImageWriter {
 public:
  virtual ~ImageWriter() = default;

  /// Initialize the codec to write to the provided `writer`.
  /// This is not done in the constructor in order to allow returning errors
  /// to the caller.
  ///
  /// \checks `reader` is not null.
  virtual absl::Status Initialize(riegeli::Writer*) = 0;

  /// Encode an image buffer into the writer.
  /// When supported, repeated calls  may encode multiple images.
  ///
  /// The source is assumed to be in C-order (height, width, num_components).
  ///
  /// \checks `Initialize` returned `absl::OkStatus()`
  virtual absl::Status Encode(
      const ImageInfo& image,
      tensorstore::span<const unsigned char> source) = 0;

  /// Finish writing. Closes the writer and returns the status.
  virtual absl::Status Done() = 0;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_IMAGE_WRITER_H_
