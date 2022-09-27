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

#ifndef TENSORSTORE_INTERNAL_IMAGE_AVIF_READER_H_
#define TENSORSTORE_INTERNAL_IMAGE_AVIF_READER_H_

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/internal/image/avif_common.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_reader.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

/// AVIF data is encoded in YUV format images. When an image is detected in
/// the following format, the mapping to channels is:
///  YUV (400) <=> 1 channel
///  YUVA (400) <=> 2 channels
///  YUV (444/422/420) <=> 3 channels (converted to RGB)
///  YUVA (444/422/420) <=> 4 channels (converted to RGBA)
///
/// The AVIF image format stores alpha as an independent greyscale channel (A),
/// so encoding single channel single pane images may be the best approach
/// for volumetric data.
///
/// Images are assumed to be uint8_t, even if they were encoded using 10 or more
/// bits.
///
/// NOTE: AVIF may have perceptual loss that is not constant across channels,
/// so before encoding volumetric panes with more than a single channel, it
/// is important to test whether this affects the dataset.
struct AvifReaderOptions {
  /// AVIF image format is YUV(A); generally assume that the input is RGB(A)
  /// Generally assume that the image source is RGB(A).
  bool convert_to_rgb = true;
};

class AvifReader : public ImageReader {
 public:
  AvifReader() = default;
  ~AvifReader() override = default;

  // Allow move.
  AvifReader(AvifReader&& src) = default;
  AvifReader& operator=(AvifReader&& src) = default;

  // Initialize the decoder.
  absl::Status Initialize(riegeli::Reader* reader) override;

  // Returns the current ImageInfo.
  ImageInfo GetImageInfo() override;

  // Decodes the next available image into 'dest'.
  absl::Status Decode(tensorstore::span<unsigned char> dest) override {
    return DecodeImpl(dest, {});
  }
  absl::Status Decode(tensorstore::span<unsigned char> dest,
                      const AvifReaderOptions& options) {
    return DecodeImpl(dest, options);
  }

 private:
  absl::Status DecodeImpl(tensorstore::span<unsigned char> dest,
                          const AvifReaderOptions& options);

  std::unique_ptr<avifDecoder, AvifDeleter> decoder_;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_AVIF_READER_H_
