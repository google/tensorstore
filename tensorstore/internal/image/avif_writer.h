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

#ifndef TENSORSTORE_INTERNAL_IMAGE_AVIF_WRITER_H_
#define TENSORSTORE_INTERNAL_IMAGE_AVIF_WRITER_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/image/avif_common.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_writer.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_image {

/// AVIF data is encoded in YUV format images. When an image is encoded,
/// depending on the number of channels, it is converted to YUV(A) as follows:
///  YUV (400) <=> 1 channel
///  YUVA (400) <=> 2 channels
///  YUV (444) <=> 3 channels (converted to RGB)
///  YUVA (444) <=> 4 channels (converted to RGBA)
///
/// The AVIF image format stores alpha as an independent greyscale channel (A),
/// so encoding single channel single pane images may be the best approach
/// for volumetric data.
///
/// NOTE: AVIF may have perceptual loss that is not constant across channels,
/// so before encoding volumetric panes with more than a single channel, it
/// is important to test whether this affects the dataset.

struct AvifWriterOptions {
  /// Quality, with ranges from 0 (lossless) to 63 (worst) inclusive.
  /// Empirical evidence suggests that a value around 45 works well.
  int quantizer = 0;

  /// Speed setting, with ranges from 0 (slowest) to (10) fastest.
  /// The default speed is a decent tradeoff between quality, cpu, and size,
  /// and values > 6 are intended to be used for real-time mode.
  /// See: https://www.biorxiv.org/content/10.1101/2021.05.29.445828v2
  ///
  /// For EM single channel data, the evidence suggests that options
  /// quality=45, speed=2 works well.
  int speed = 6;

  /// AVIF stores images as YUV(A); it can convert from RGB(A) when necessary.
  bool input_is_rgb = true;
};

class AvifWriter : public ImageWriter {
 public:
  AvifWriter() = default;
  ~AvifWriter() override = default;

  // Allow move.
  AvifWriter(AvifWriter&& src) = default;
  AvifWriter& operator=(AvifWriter&& src) = default;

  // Initialize the codec. This is not done in the constructor in order
  // to allow returning errors to the caller.
  absl::Status Initialize(riegeli::Writer* writer) override {
    return InitializeImpl(std::move(writer), {});
  }
  absl::Status Initialize(riegeli::Writer* writer,
                          const AvifWriterOptions& options) {
    return InitializeImpl(std::move(writer), options);
  }

  // Encodes image with the provided options.
  absl::Status Encode(const ImageInfo& info,
                      tensorstore::span<const unsigned char> source) override;

  /// Finish writing. Closes the writer and returns the status.
  absl::Status Done() override;

 private:
  absl::Status InitializeImpl(riegeli::Writer* writer,
                              const AvifWriterOptions& options);

  riegeli::Writer* writer_ = nullptr;  // unowned
  AvifWriterOptions options_;
  std::unique_ptr<avifEncoder, AvifDeleter> encoder_;
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_AVIF_WRITER_H_
