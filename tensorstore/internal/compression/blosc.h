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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorstore/util/status.h"

/// Convenience interface to the blosc library.

namespace tensorstore {
namespace blosc {

/// Specifies the Blosc encode options.
///
/// Refer to the blosc library `blosc_compress_ctx` function documentation for
/// details.
struct Options {
  /// Must be one of the supported compressor names.
  ///
  /// The list of supported compressors (determined by the build configuration)
  /// are returned by the `blosc_list_compressors` function.
  const char* compressor;

  /// Specifies the desired compression level, must be in the range `[0, 9]`,
  /// where `0` indicates no compression and `9` indicates maximum compression.
  int clevel;

  /// Must be one of `BLOSC_NOSHUFFLE` (no shuffling), `BLOSC_SHUFFLE`
  /// (byte-wise shuffling), `BLOSC_BITSHUFFLE` (bit-wise shuffling), or the
  /// special value of `-1`, which is equivalent to `BLOSC_BITSHUFFLE` if
  /// `element_size == 1`, otherwise is equivalent to `BLOSC_SHUFFLE`.
  int shuffle;

  /// Requested size of block into which to divide the input before passing to
  /// the underlying compressor.  The specified value is a hint and may be
  /// ignored.
  std::size_t blocksize;

  /// Specifies that `input` is a sequence of elements of `element_size` bytes.
  /// This only affects shuffling.
  std::size_t element_size;
};

/// Compresses `input` and append the result to `*output`.
///
/// \param input The input data to compress.
/// \param output[in,out] Output cord to which compressed data will be appended.
/// \param options Specifies compression options.
/// \error `absl::StatusCode::kInvalidArgument` if `input.size()` exceeds
///     `BLOSC_MAX_BUFFERSIZE`.
Status Encode(const absl::Cord& input, absl::Cord* output,
              const Options& options);

/// Decompresses `input` and append the result to `*output`.
///
/// \param input The input data to decompress.
/// \param output[in,out] Output cord to which decompressed data will be
///     appended.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Status Decode(const absl::Cord& input, absl::Cord* output);

}  // namespace blosc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_
