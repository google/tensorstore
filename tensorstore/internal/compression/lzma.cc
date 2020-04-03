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

#include "tensorstore/internal/compression/lzma.h"

#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace lzma {

::lzma_ret BufferManager::Process() {
  ::lzma_ret r;
  do {
    strm.avail_out = kBufferSize;
    strm.next_out = reinterpret_cast<std::uint8_t*>(&buffer[0]);
    r = ::lzma_code(&strm, LZMA_FINISH);
    output->insert(output->end(), buffer,
                   buffer + kBufferSize - strm.avail_out);
  } while (r == LZMA_OK);
  return r;
}

Status GetInitErrorStatus(::lzma_ret r) {
  switch (r) {
    case LZMA_OK:
      return absl::OkStatus();
    case LZMA_MEM_ERROR:
    case LZMA_OPTIONS_ERROR:
    case LZMA_UNSUPPORTED_CHECK:
    case LZMA_PROG_ERROR:
    default:
      TENSORSTORE_CHECK(false);
  }
}

Status GetEncodeErrorStatus(::lzma_ret r) {
  switch (r) {
    case LZMA_STREAM_END:
      return absl::OkStatus();
    case LZMA_DATA_ERROR:
      return absl::InvalidArgumentError("Maximum LZMA data size exceeded");
    case LZMA_MEMLIMIT_ERROR:
    case LZMA_MEM_ERROR:
    default:
      TENSORSTORE_CHECK(false);
  }
}

Status GetDecodeErrorStatus(::lzma_ret r) {
  switch (r) {
    case LZMA_STREAM_END:
      return absl::OkStatus();
    case LZMA_FORMAT_ERROR:
      return absl::InvalidArgumentError("LZMA format not recognized");
    case LZMA_OPTIONS_ERROR:
      return absl::InvalidArgumentError("Unsupported LZMA options");
    case LZMA_DATA_ERROR:
    case LZMA_BUF_ERROR:
      return absl::InvalidArgumentError("LZMA-encoded data is corrupt");
    case LZMA_MEM_ERROR:
    case LZMA_MEMLIMIT_ERROR:
    default:
      TENSORSTORE_CHECK(false);
  }
}

namespace xz {
Status Encode(absl::string_view input, std::string* output, Options options) {
  lzma::BufferManager manager(input, output);
  ::lzma_ret err =
      ::lzma_easy_encoder(&manager.strm, options.preset, options.check);
  if (err != LZMA_OK) return lzma::GetInitErrorStatus(err);
  // element_size is not used for xz compression.
  return lzma::GetEncodeErrorStatus(manager.Process());
}
Status Decode(absl::string_view input, std::string* output) {
  lzma::BufferManager manager(input, output);
  ::lzma_ret err = ::lzma_stream_decoder(
      &manager.strm, /*memlimit=*/std::numeric_limits<std::uint64_t>::max(),
      /*flags=*/0);
  if (err != LZMA_OK) return lzma::GetInitErrorStatus(err);
  // element_size is not used for xz compression.
  return lzma::GetDecodeErrorStatus(manager.Process());
}
}  // namespace xz

}  // namespace lzma
}  // namespace tensorstore
