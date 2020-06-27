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

#include "tensorstore/internal/compression/zlib.h"

#include "tensorstore/internal/compression/cord_stream_manager.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/status.h"

// Include zlib header last because it defines a bunch of poorly-named macros.
#include <zlib.h>

namespace tensorstore {
namespace zlib {
namespace {

struct InflateOp {
  static int Init(z_stream* s, [[maybe_unused]] int level, int header_option) {
    return inflateInit2(s, /*windowBits=*/15 /* (default) */
                               + header_option);
  }
  static int Process(z_stream* s, int flags) { return inflate(s, flags); }
  static int Destroy(z_stream* s) { return inflateEnd(s); }
  static constexpr bool kDataErrorPossible = true;
};

struct DeflateOp {
  static int Init(z_stream* s, int level, int header_option) {
    return deflateInit2(s, level, Z_DEFLATED,
                        /*windowBits=*/15 /* (default) */
                            + header_option,
                        /*memlevel=*/8 /* (default) */,
                        /*strategy=*/Z_DEFAULT_STRATEGY);
  }
  static int Process(z_stream* s, int flags) { return deflate(s, flags); }
  static int Destroy(z_stream* s) { return deflateEnd(s); }
  static constexpr bool kDataErrorPossible = false;
};

/// Inflates or deflates using zlib.
///
/// \tparam Op Either `InflateOp` or `DeflateOp`.
/// \param input The input data.
/// \param output[in,out] Output buffer to which output data is appended.
/// \param level Compression level, must be in the range [0, 9].
/// \param use_gzip_header If `true`, use gzip header.  Otherwise, use zlib
///     header.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if decoding fails due to input
///     input.
template <typename Op>
Status ProcessZlib(const absl::Cord& input, absl::Cord* output, int level,
                   bool use_gzip_header) {
  z_stream s = {};
  internal::CordStreamManager<z_stream, /*BufferSize=*/16 * 1024>
      stream_manager(s, input, output);
  const int header_option = use_gzip_header ? 16 /* require gzip header */
                                            : 0;
  int err = Op::Init(&s, level, header_option);
  if (err != Z_OK) {
    // Terminate if allocating even the small amount of memory required fails.
    TENSORSTORE_CHECK(false);
  }
  struct StreamDestroyer {
    z_stream* s;
    ~StreamDestroyer() { Op::Destroy(s); }
  } stream_destroyer{&s};

  while (true) {
    const bool input_complete = stream_manager.FeedInputAndOutputBuffers();
    err = Op::Process(&s, input_complete ? Z_FINISH : Z_NO_FLUSH);
    const bool made_progress = stream_manager.HandleOutput();
    if (err == Z_OK) continue;
    if (err == Z_BUF_ERROR && made_progress) continue;
    break;
  }
  switch (err) {
    case Z_STREAM_END:
      if (!stream_manager.has_input_remaining()) {
        return absl::OkStatus();
      }
      [[fallthrough]];
    case Z_NEED_DICT:
    case Z_DATA_ERROR:
    case Z_BUF_ERROR:
      if (!Op::kDataErrorPossible) {
        TENSORSTORE_CHECK(false);
      }
      return absl::InvalidArgumentError("Error decoding zlib-compressed data");
    default:
      TENSORSTORE_CHECK(false);
  }
}

}  // namespace

void Encode(const absl::Cord& input, absl::Cord* output,
            const Options& options) {
  ProcessZlib<DeflateOp>(input, output, options.level, options.use_gzip_header)
      .IgnoreError();
}

Status Decode(const absl::Cord& input, absl::Cord* output,
              bool use_gzip_header) {
  return ProcessZlib<InflateOp>(input, output, 0, use_gzip_header);
}

}  // namespace zlib
}  // namespace tensorstore
