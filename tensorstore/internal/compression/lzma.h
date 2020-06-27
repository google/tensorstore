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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_LZMA_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_LZMA_H_

/// \file
/// Convenience interface to the lzma library.

#include <cstddef>
#include <cstdint>

#include "absl/strings/cord.h"
#include <lzma.h>
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace lzma {
namespace xz {

/// Options for the XZ format, which is a variant of LZMA with simplified
/// options.
struct Options {
  /// Specifies compression level.  Must be a number in `[0, 9]`, optionally ORd
  /// with `LZMA_PRESET_EXTREME` to provide marginal improvements in compression
  /// ratio at the cost of slower encoding.
  std::uint32_t preset = 6;

  /// Specifies the integrity check to use.
  ::lzma_check check = LZMA_CHECK_CRC64;
};

/// Compresses `input` and appends the result to `*output`.
///
/// \param input Input to encode.
/// \param output[in,out] Output cord to which compressed data will be appended.
/// \param options Specifies the compression options.
Status Encode(const absl::Cord& input, absl::Cord* output, Options options);

/// Decompresses `input` and appends the result to `*output`.
///
/// \param input Input to decode.
/// \param output[in,out] Output cord to which decompressed data will be
///     appended.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Status Decode(const absl::Cord& input, absl::Cord* output);

}  // namespace xz
}  // namespace lzma
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_LZMA_H_
