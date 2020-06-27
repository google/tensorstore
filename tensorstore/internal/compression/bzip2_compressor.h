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

#include <cstddef>
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/internal/compression/bzip2.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"

namespace tensorstore {
namespace internal {

class Bzip2Compressor : public internal::JsonSpecifiedCompressor,
                        public bzip2::Options {
 public:
  Status Encode(const absl::Cord& input, absl::Cord* output,
                std::size_t element_size) const override {
    // element_size is not used for bzip2 compression.
    bzip2::Encode(input, output, *this);
    return absl::OkStatus();
  }
  Status Decode(const absl::Cord& input, absl::Cord* output,
                std::size_t element_size) const override {
    // element_size is not used for bzip2 compression.
    return bzip2::Decode(input, output);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BZIP2_COMPRESSOR_H_
