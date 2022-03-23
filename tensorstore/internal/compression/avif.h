// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_AVIF_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_AVIF_H_

#include <stddef.h>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace avif {

/// AVIF data is encoded in YUV format images. When an image is detected in
/// the following format, the mapping to channels is:
///  YUV (400) <=> 1 channel
///  YUVA (400) <=> 2 channels
///  YUV (444/422/420) <=> 3 channels (converted to RGB)
///  YUVA (444/422/420) <=> 4 channels (converted to RGBA)
///
/// The AVIF image format stores alpha as an independent greyscale channel,
/// so encoding single channel single pane images may be the best approach
/// for volumetric data.
///
/// NOTE: AVIF may have perceptual loss that is not constant across channels,
/// so before encoding volumetric panes with more than a single channel, it
/// is important to test whether this affects the dataset.

/// Decodes an AVIF-format `source` buffer.
///
/// \param source The source data.
/// \param allocate_buffer Callback invoked once the header is successfully
///     parsed.  May return an error to indicate that the dimensions are not
///     acceptable.  Otherwise, must return a non-null pointer to a buffer of
///     size `width * height * num_components` that will be treated as a C order
///     `(height, width, num_components)` array in which to store the decoded
///     image.
/// \returns `OkStatus()` on success, or the error returned by `validate_size`.
/// \error `absl::StatusCode::kInvalidArgument` if the source data is corrupt.
absl::Status Decode(const absl::Cord& input,
                    absl::FunctionRef<Result<unsigned char*>(
                        size_t width, size_t height, size_t num_components)>
                        allocate_buffer);

struct EncodeOptions {
  /// Quality, with ranges from 0 (lossless) to 63 (worst) inclusive.
  /// Empirical evidence suggests that a value around 45 works well.
  int quantizer = 0;
  /// Speed setting, with ranges from 0 (slowest) to (10) fastest.
  /// The default speed is a decent tradeoff between quality, cpu, and size,
  /// and values > 6 are intended to be used for real-time mode.
  int speed = 6;

  /// See: https://www.biorxiv.org/content/10.1101/2021.05.29.445828v2
  ///
  /// For EM single channel data, the evidence suggests that options
  /// quality=45, speed=2 works well.
  ///
  /// NOTE: Consider an option to expose whether to encode as RBG or YUV.
};

/// Encodes a source array in AVIF-format.
///
/// \param source Pointer to C order `(height, width, num_components)` array
///     containing the image data.
/// \param width Image width
/// \param height Image height
/// \param num_components Number of components, may be 1,2,3, or 4.
/// \param options Encoding options.
/// \param output[out] Output buffer to which encoded PNG will be appended.
/// \error `absl::StatusCode::kInvalidArgument` if the source image array has
///     invalid dimensions.
absl::Status Encode(const unsigned char* source, size_t width, size_t height,
                    size_t num_components, const EncodeOptions& options,
                    absl::Cord* output);

}  // namespace avif
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_AVIF_H_
