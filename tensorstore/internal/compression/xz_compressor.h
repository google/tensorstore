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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_XZ_COMPRESSOR_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_XZ_COMPRESSOR_H_

/// \file Define an XZ-format JsonSpecifiedCompressor.

#include <cstddef>

#include <lzma.h>
#include "tensorstore/internal/compression/json_specified_compressor.h"

namespace tensorstore {
namespace internal {

/// Options for the XZ format, which is a variant of LZMA with simplified
/// options.
struct XzOptions {
  /// Specifies compression level.  Must be a number in `[0, 9]`.
  int level = 6;

  /// Specifies whether to use LZMA_PRESET_EXTREME, to provide marginal
  /// improvements in compression ratio at the cost of slower encoding.
  bool extreme = false;

  /// Specifies the integrity check to use.
  ::lzma_check check = LZMA_CHECK_CRC64;
};

class XzCompressor : public JsonSpecifiedCompressor, public XzOptions {
 public:
  std::unique_ptr<riegeli::Writer> GetWriter(
      std::unique_ptr<riegeli::Writer> base_writer,
      size_t element_bytes) const override;

  std::unique_ptr<riegeli::Reader> GetReader(
      std::unique_ptr<riegeli::Reader> base_reader,
      size_t element_bytes) const override;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_XZ_COMPRESSOR_H_
