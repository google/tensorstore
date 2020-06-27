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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_JPEG_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_JPEG_H_

#include "absl/strings/cord.h"
#include "tensorstore/util/function_view.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace jpeg {

struct EncodeOptions {
  /// Specifies the JPEG quality, per the IJG (Independent JPEG Group) [0, 100]
  /// recommended scale, with 0 being the worst quality (smallest file size) and
  /// 100 the best quality (largest file size).
  int quality = 75;
};

/// Decodes a JPEG-format `source` buffer.
///
/// \param source The source data.
/// \param validate_size Callback invoked once the header is successfully
///     parsed.  May return an error to indicate that the dimensions are not
///     acceptable.  Otherwise, must return a non-null pointer to a buffer of
///     size `width * height * num_components` that will be treated as a C order
///     `(height, width, num_components)` array in which to store the decoded
///     image.
/// \returns `Status()` on success, or the error returned by `validate_size`.
/// \error `absl::StatusCode::kInvalidArgument` if the source data is corrupt.
Status Decode(const absl::Cord& input,
              FunctionView<Result<unsigned char*>(size_t width, size_t height,
                                                  size_t num_components)>
                  validate_size);

/// Encodes a source array in JPEG format.
///
/// \param source Pointer to C order `(height, width, num_components)` array
///     containing the image data.
/// \param width Image width
/// \param height Image height
/// \param num_components Number of components, must be 1 or 3.
/// \param options Encoding options.
/// \param output[out] Output buffer to which encoded JPEG will be appended.
/// \error `absl::StatusCode::kInvalidArgument` if the source image array has
///     invalid dimensions.
Status Encode(const unsigned char* source, size_t width, size_t height,
              size_t num_components, const EncodeOptions& options,
              absl::Cord* output);

}  // namespace jpeg
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_JPEG_H_
