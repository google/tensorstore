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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_

/// Convenience interface to the blosc library.

#include <stddef.h>

#include <memory>
#include <string>
#include <string_view>

#include "absl/functional/function_ref.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace blosc {

/// Specifies the Blosc encode options.
///
/// Refer to the blosc library `blosc_compress_ctx` function documentation for
/// details.
struct Options {
  /// Must be one of the supported compressor names.
  ///
  /// The list of supported compressors (determined by the build configuration)
  /// are returned by the `blosc_list_compressors` function.
  const char* compressor;

  /// Specifies the desired compression level, must be in the range `[0, 9]`,
  /// where `0` indicates no compression and `9` indicates maximum compression.
  int clevel;

  /// Must be one of `BLOSC_NOSHUFFLE` (no shuffling), `BLOSC_SHUFFLE`
  /// (byte-wise shuffling), `BLOSC_BITSHUFFLE` (bitwise shuffling), or the
  /// special value of `-1`, which is equivalent to `BLOSC_BITSHUFFLE` if
  /// `element_size == 1`, otherwise is equivalent to `BLOSC_SHUFFLE`.
  int shuffle;

  /// Requested size of block into which to divide the input before passing to
  /// the underlying compressor.  The specified value is a hint and may be
  /// ignored.
  size_t blocksize;

  /// Specifies that `input` is a sequence of elements of `element_size` bytes.
  /// This only affects shuffling.
  size_t element_size;
};

/// Compresses `input`.
///
/// \param input The input data to compress.
/// \param options Specifies compression options.
/// \error `absl::StatusCode::kInvalidArgument` if `input.size()` exceeds
///     `BLOSC_MAX_BUFFERSIZE`.
Result<std::string> Encode(std::string_view input, const Options& options);

// Same as above, but calls `get_output_buffer` to obtain a sufficiently long
// output buffer, and then returns the actual size of the encoded data. If
// `get_output_buffer` returns `nullptr`, then encoding is skipped and `0` is
// returned. The encoded size is less than or equal to the size passed to
// `get_output_buffer`.
Result<size_t> EncodeWithCallback(
    std::string_view input, const Options& options,
    absl::FunctionRef<char*(size_t)> get_output_buffer);

/// Decompresses `input`.
///
/// \param input The input data to decompress.
/// \error `absl::StatusCode::kInvalidArgument` if `input` is corrupt.
Result<std::string> Decode(std::string_view input);

// Same as above, but calls `get_output_buffer` to obtain the output buffer, and
// returns the size of the decoded data. If `get_output_buffer` returns
// `nullptr`, then decoding is skipped and `0` is returned. On success, the
// decoded size is always equal to the size passed to `get_output_buffer`.
Result<size_t> DecodeWithCallback(
    std::string_view input, absl::FunctionRef<char*(size_t)> get_output_buffer);

// Returns the decoded size of the input.
Result<size_t> GetDecodedSize(std::string_view input);

// Writes blosc-encoded data to an underlying writer.
//
// Because the c-blosc library does not support streaming, this buffers the
// entire decoded value.
//
// TODO(jbms): Change this to avoid copying if the source data is already in
// a single flat buffer but cannot be converted to a Cord or Chain due to
// ownership/lifetime limitations.
class BloscWriter : public riegeli::CordWriter<absl::Cord> {
 public:
  explicit BloscWriter(const blosc::Options& options,
                       riegeli::Writer& base_writer);

  void Done() override;

 private:
  blosc::Options options_;
  riegeli::Writer& base_writer_;
};

// Reads blosc-encoded data from an underlying reader.
//
// Because the c-blosc library does not support streaming, this buffers the
// entire encoded value.
class BloscReader : public riegeli::Reader {
 public:
  explicit BloscReader(riegeli::Reader& base_reader);
  BloscReader(BloscReader&&) = delete;
  bool ToleratesReadingAhead() override;
  bool SupportsSize() override;

 protected:
  bool PullSlow(size_t min_length, size_t recommended_length) override;
  bool ReadSlow(size_t length, char* dest) override;
  absl::optional<riegeli::Position> SizeImpl() override {
    return decoded_size_;
  }

 private:
  riegeli::Reader& base_reader_;
  absl::string_view encoded_data_;
  size_t decoded_size_;
  std::unique_ptr<char[]> buffer_;
};

}  // namespace blosc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_H_
