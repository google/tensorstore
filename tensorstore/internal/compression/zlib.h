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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_ZLIB_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_ZLIB_H_

/// \file
/// Convenience interface to the zlib library.

#include <cstddef>
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace zlib {

struct Options {
  /// Specifies the compression level, must be in the range `[-1, 9]`, with `0`
  /// being no compression and `9` being the most compression.  The special
  /// value `-1` indicates the zlib default compression, which is equivalent to
  /// 6.
  int level = -1;

  /// Specifies whether to use the gzip header rather than the zlib header
  /// format.
  bool use_gzip_header = false;
};

/// Compresses `input` and appends the result to `*output`.
///
/// \param input Input to encode.
/// \param output[in,out] Output cord to which compressed data will be appended.
/// \param options Specifies the compression options.
void Encode(const absl::Cord& input, absl::Cord* output,
            const Options& options);

/// Decompresses `input` and appends the result to `*output`.
///
/// \param input Input to decode.
/// \param output[in,out] Output cord to which decompressed data will be
///     appended.
/// \param use_gzip_header Specifies the header type with which `input` was
///     encoded.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
absl::Status Decode(const absl::Cord& input, absl::Cord* output,
                    bool use_gzip_header);

}  // namespace zlib
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_ZLIB_H_
