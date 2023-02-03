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

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "tensorstore/internal/compression/cord_stream_manager.h"

// Include this last since it defines many macros.
#include <bzlib.h>

namespace tensorstore {
namespace bzip2 {

namespace {

using StreamManager =
    internal::CordStreamManager<bz_stream, /*BufferSize=*/16 * 1024>;

}  // namespace

void Encode(const absl::Cord& input, absl::Cord* output,
            const Options& options) {
  assert(options.block_size_100k >= 1 && options.block_size_100k <= 9);
  bz_stream stream = {};
  StreamManager stream_manager(stream, input, output);
  int err = BZ2_bzCompressInit(&stream, options.block_size_100k,
                               /*verbosity=*/0,
                               // Default work factor, which affects performance
                               // but not the compressed result.
                               /*workFactor=*/0);
  ABSL_CHECK_EQ(err, BZ_OK);
  struct StreamWrapper {
    bz_stream* stream;
    ~StreamWrapper() { BZ2_bzCompressEnd(stream); }
  } stream_wrapper{&stream};
  do {
    const bool input_complete = stream_manager.FeedInputAndOutputBuffers();
    err = BZ2_bzCompress(&stream, input_complete ? BZ_FINISH : BZ_RUN);
    stream_manager.HandleOutput();
  } while (err == BZ_RUN_OK || err == BZ_FINISH_OK);
  switch (err) {
    case BZ_STREAM_END:
      return;
    case BZ_SEQUENCE_ERROR:
    default:
      ABSL_CHECK(false);
  }
}

absl::Status Decode(const absl::Cord& input, absl::Cord* output) {
  bz_stream stream = {};
  StreamManager stream_manager(stream, input, output);
  int err = BZ2_bzDecompressInit(&stream, /*verbosity=*/0,
                                 // No need to reduce memory usage.
                                 /*small=*/0);
  ABSL_CHECK_EQ(err, BZ_OK);
  struct StreamWrapper {
    bz_stream* stream;
    ~StreamWrapper() { BZ2_bzDecompressEnd(stream); }
  } stream_wrapper{&stream};
  while (true) {
    stream_manager.FeedInputAndOutputBuffers();
    err = BZ2_bzDecompress(&stream);
    const bool made_progress = stream_manager.HandleOutput();
    if (err != BZ_OK ||
        // In the case of corrupted input, BZ_STREAM_END may never be returned
        // even though all input has been consumed.
        !made_progress) {
      break;
    }
  }
  switch (err) {
    case BZ_STREAM_END:
      if (!stream_manager.has_input_remaining()) {
        return absl::OkStatus();
      }
      [[fallthrough]];
    case BZ_OK:
    case BZ_DATA_ERROR_MAGIC:
    case BZ_DATA_ERROR:
      return absl::InvalidArgumentError("Error decoding bzip2-compressed data");
    case BZ_MEM_ERROR:
    case BZ_PARAM_ERROR:
    default:
      ABSL_CHECK(false);
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

}  // namespace bzip2
}  // namespace tensorstore
