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

#include "tensorstore/internal/compression/cord_stream_manager.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace lzma {

/// RAII adapter for LZMA stream encoding/decoding.
struct BufferManager {
  constexpr static std::size_t kBufferSize = 16 * 1024;
  lzma_stream stream = LZMA_STREAM_INIT;

  explicit BufferManager(const absl::Cord& input, absl::Cord* output)
      : stream_manager_(stream, input, output) {}

  ::lzma_ret Process();

  ~BufferManager() { ::lzma_end(&stream); }

 private:
  internal::CordStreamManager<lzma_stream, kBufferSize> stream_manager_;
};

::lzma_ret BufferManager::Process() {
  ::lzma_ret r;
  do {
    const bool input_complete = stream_manager_.FeedInputAndOutputBuffers();
    r = ::lzma_code(&stream, input_complete ? LZMA_FINISH : LZMA_RUN);
    stream_manager_.HandleOutput();
  } while (r == LZMA_OK);
  return r;
}

/// Returns the Status associated with a liblzma error.
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
Status Encode(const absl::Cord& input, absl::Cord* output, Options options) {
  lzma::BufferManager manager(input, output);
  ::lzma_ret err =
      ::lzma_easy_encoder(&manager.stream, options.preset, options.check);
  if (err != LZMA_OK) return lzma::GetInitErrorStatus(err);
  return lzma::GetEncodeErrorStatus(manager.Process());
}
Status Decode(const absl::Cord& input, absl::Cord* output) {
  lzma::BufferManager manager(input, output);
  ::lzma_ret err = ::lzma_stream_decoder(
      &manager.stream, /*memlimit=*/std::numeric_limits<std::uint64_t>::max(),
      /*flags=*/0);
  if (err != LZMA_OK) return lzma::GetInitErrorStatus(err);
  return lzma::GetDecodeErrorStatus(manager.Process());
}
}  // namespace xz

}  // namespace lzma
}  // namespace tensorstore
