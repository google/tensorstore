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

#include "tensorstore/internal/compression/blosc.h"

#include <blosc.h>
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace blosc {

Status Encode(absl::string_view input, std::string* output,
              const Options& options) {
  if (input.size() > BLOSC_MAX_BUFFERSIZE) {
    return absl::InvalidArgumentError(
        StrCat("Blosc compression input of ", input.size(),
               " bytes exceeds maximum size of ", BLOSC_MAX_BUFFERSIZE));
  }

  const size_t output_offset = output->size();
  const size_t output_bound = input.size() + BLOSC_MAX_OVERHEAD;
  output->resize(output_offset + output_bound);
  int shuffle = options.shuffle;
  if (shuffle == -1) {
    shuffle = options.element_size == 1 ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE;
  }
  int n = blosc_compress_ctx(options.clevel, shuffle, options.element_size,
                             input.size(), input.data(),
                             output->data() + output_offset, output_bound,
                             options.compressor, options.blocksize,
                             /*numinternalthreads=*/1);
  if (n < 0) {
    return absl::InternalError(StrCat("Internal blosc error: ", n));
  }
  output->resize(output_offset + n);
  return absl::OkStatus();
}

Status Decode(absl::string_view input, std::string* output) {
  size_t nbytes;
  if (blosc_cbuffer_validate(input.data(), input.size(), &nbytes) != 0) {
    return absl::InvalidArgumentError("Invalid blosc-compressed data");
  }
  const std::size_t output_offset = output->size();
  output->resize(output_offset + nbytes);
  if (nbytes == 0) return absl::OkStatus();
  const int n =
      blosc_decompress_ctx(input.data(), output->data() + output_offset, nbytes,
                           /*numinternalthreads=*/1);
  if (n <= 0) {
    return absl::InvalidArgumentError(StrCat("Blosc error: ", n));
  }
  return absl::OkStatus();
}

}  // namespace blosc
}  // namespace tensorstore
