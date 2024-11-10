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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_COMPRESSOR_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_COMPRESSOR_H_

/// \file Defines a bzip2 JsonSpecifiedCompressor.

#include <stddef.h>

#include <memory>

#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"

namespace tensorstore {
namespace internal {

struct Bzip2Options {
  /// Determines the block size, which affects the compression level and memory
  /// usage.  Must be in the range `[1, 9]`.  The actual block size is
  /// `100000 * level`.
  int level = 1;
};

class Bzip2Compressor : public internal::JsonSpecifiedCompressor,
                        public Bzip2Options {
 public:
  std::unique_ptr<riegeli::Writer> GetWriter(
      riegeli::Writer& base_writer, size_t element_bytes) const override;

  virtual std::unique_ptr<riegeli::Reader> GetReader(
      riegeli::Reader& base_reader, size_t element_bytes) const override;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_COMPRESSOR_H_
