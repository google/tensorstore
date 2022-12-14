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

#include "absl/status/status.h"
#include <blosc.h>
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace blosc {

absl::Status Encode(const absl::Cord& input, absl::Cord* output,
                    const Options& options) {
  if (input.size() > BLOSC_MAX_BUFFERSIZE) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Blosc compression input of ", input.size(),
        " bytes exceeds maximum size of ", BLOSC_MAX_BUFFERSIZE));
  }
  // Blosc requires a contiguous input and output buffer.
  absl::Cord input_copy(input);
  auto input_flat = input_copy.Flatten();
  internal::FlatCordBuilder output_buffer(input.size() + BLOSC_MAX_OVERHEAD);
  int shuffle = options.shuffle;
  if (shuffle == -1) {
    shuffle = options.element_size == 1 ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE;
  }
  int n = blosc_compress_ctx(options.clevel, shuffle, options.element_size,
                             input_flat.size(), input_flat.data(),
                             output_buffer.data(), output_buffer.size(),
                             options.compressor, options.blocksize,
                             /*numinternalthreads=*/1);
  if (n < 0) {
    return absl::InternalError(
        tensorstore::StrCat("Internal blosc error: ", n));
  }
  output_buffer.resize(n);
  output->Append(std::move(output_buffer).Build());
  return absl::OkStatus();
}

absl::Status Decode(const absl::Cord& input, absl::Cord* output) {
  size_t nbytes;
  // Blosc requires a contiguous input and output buffer.
  absl::Cord input_copy(input);
  auto input_flat = input_copy.Flatten();
  if (blosc_cbuffer_validate(input_flat.data(), input_flat.size(), &nbytes) !=
      0) {
    return absl::InvalidArgumentError("Invalid blosc-compressed data");
  }
  internal::FlatCordBuilder output_buffer(nbytes);
  if (nbytes == 0) return absl::OkStatus();
  const int n =
      blosc_decompress_ctx(input_flat.data(), output_buffer.data(), nbytes,
                           /*numinternalthreads=*/1);
  if (n <= 0) {
    return absl::InvalidArgumentError(tensorstore::StrCat("Blosc error: ", n));
  }
  output->Append(std::move(output_buffer).Build());
  return absl::OkStatus();
}

}  // namespace blosc
}  // namespace tensorstore
