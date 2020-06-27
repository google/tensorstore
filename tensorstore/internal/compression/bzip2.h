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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_H_

#include <cstddef>
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/util/status.h"

/// Convenience interface to the bzip2 library.

namespace tensorstore {
namespace bzip2 {

struct Options {
  /// Determines the block size, which affects the compression level and memory
  /// usage.  Must be in the range `[1, 9]`.  The actual block size is
  /// `100000 * level`.
  int block_size_100k;
};

/// Compresses `input` and append the result to `*output`.
///
/// \param input The input data to compress.
/// \param output[in,out] Output cord to which compressed data will be appended.
/// \param options Specifies compression options.
void Encode(const absl::Cord& input, absl::Cord* output,
            const Options& options);

/// Decompresses `input` and append the result to `*output`.
///
/// \param input The input data to decompress.
/// \param output[in,out] Output cord to which decompressed data will be
///     appended.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Status Decode(const absl::Cord& input, absl::Cord* output);

}  // namespace bzip2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_H_
