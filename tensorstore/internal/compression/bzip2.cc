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

#include "tensorstore/internal/compression/bzip2.h"

#include "tensorstore/util/status.h"

// Include this last since it defines many macros.
#include <bzlib.h>

namespace tensorstore {
namespace bzip2 {

namespace {

struct BufferManager {
  absl::string_view input;
  std::string* output;

  constexpr static std::size_t kBufferSize = 16 * 1024;
  char buffer[kBufferSize];
  bz_stream strm = {};

  std::size_t total_in() const {
    return strm.total_in_lo32 +
           (static_cast<std::size_t>(strm.total_in_hi32) << 32);
  }

  /// Returns `true` if all input has been supplied.
  bool PrepareInput() {
    static_assert(sizeof(std::size_t) >= 8);
    std::size_t total_in = this->total_in();
    strm.next_in = const_cast<char*>(input.data()) + total_in;
    std::size_t max_avail = input.size() - total_in;
    std::size_t cur_avail = std::min(
        static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        max_avail);
    strm.avail_in = static_cast<unsigned int>(cur_avail);
    strm.next_out = buffer;
    strm.avail_out = kBufferSize;
    return cur_avail == max_avail;
  }

  void CopyOutput() {
    output->insert(output->end(), buffer,
                   buffer + kBufferSize - strm.avail_out);
  }
};
}  // namespace

void Encode(absl::string_view input, std::string* output,
            const Options& options) {
  assert(options.block_size_100k >= 1 && options.block_size_100k <= 9);
  BufferManager manager{input, output};
  int err = BZ2_bzCompressInit(&manager.strm, options.block_size_100k,
                               /*verbosity=*/0,
                               // Default work factor, which affects performance
                               // but not the compressed result.
                               /*workFactor=*/0);
  TENSORSTORE_CHECK(err == BZ_OK);
  struct StreamWrapper {
    bz_stream* strm;
    ~StreamWrapper() { BZ2_bzCompressEnd(strm); }
  } stream_wrapper{&manager.strm};
  do {
    const bool input_complete = manager.PrepareInput();
    err = BZ2_bzCompress(&manager.strm, input_complete ? BZ_FINISH : BZ_RUN);
    manager.CopyOutput();
  } while (err == BZ_RUN_OK || err == BZ_FINISH_OK);
  switch (err) {
    case BZ_STREAM_END:
      return;
    case BZ_SEQUENCE_ERROR:
    default:
      TENSORSTORE_CHECK(false);
  }
}

Status Decode(absl::string_view input, std::string* output) {
  BufferManager manager{input, output};
  int err = BZ2_bzDecompressInit(&manager.strm, /*verbosity=*/0,
                                 // No need to reduce memory usage.
                                 /*small=*/0);
  TENSORSTORE_CHECK(err == BZ_OK);
  struct StreamWrapper {
    bz_stream* strm;
    ~StreamWrapper() { BZ2_bzDecompressEnd(strm); }
  } stream_wrapper{&manager.strm};
  do {
    manager.PrepareInput();
    err = BZ2_bzDecompress(&manager.strm);
    manager.CopyOutput();
  } while (err == BZ_OK &&
           // In the case of corrupted input, BZ_STREAM_END may never be
           // returned even though all input has been consumed.
           (manager.total_in() != input.size() || manager.strm.avail_out == 0));
  switch (err) {
    case BZ_STREAM_END:
      return absl::OkStatus();
    case BZ_OK:
    case BZ_DATA_ERROR_MAGIC:
    case BZ_DATA_ERROR:
      return absl::InvalidArgumentError("Error decoding bzip2-compressed data");
    case BZ_MEM_ERROR:
    case BZ_PARAM_ERROR:
    default:
      TENSORSTORE_CHECK(false);
  }
}

}  // namespace bzip2
}  // namespace tensorstore
