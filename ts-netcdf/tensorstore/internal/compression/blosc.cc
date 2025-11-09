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

#include <stddef.h>

#include <cassert>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include <blosc.h>
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace blosc {

Result<std::string> Encode(std::string_view input, const Options& options) {
  std::string output;
  auto result =
      EncodeWithCallback(input, options, [&](size_t output_buffer_size) {
        output.resize(output_buffer_size);
        return output.data();
      });
  if (!result.ok()) return result.status();
  output.erase(*result);
  return output;
}

Result<size_t> EncodeWithCallback(
    std::string_view input, const Options& options,
    absl::FunctionRef<char*(size_t)> get_output_buffer) {
  if (input.size() > BLOSC_MAX_BUFFERSIZE) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Blosc compression input of ", input.size(),
        " bytes exceeds maximum size of ", BLOSC_MAX_BUFFERSIZE));
  }
  size_t output_buffer_size = input.size() + BLOSC_MAX_OVERHEAD;
  char* output_buffer = get_output_buffer(output_buffer_size);
  if (!output_buffer) return 0;
  int shuffle = options.shuffle;
  if (shuffle == -1) {
    shuffle = options.element_size == 1 ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE;
  }
  const int n = blosc_compress_ctx(
      options.clevel, shuffle, options.element_size, input.size(), input.data(),
      output_buffer, output_buffer_size, options.compressor, options.blocksize,
      /*numinternalthreads=*/1);
  if (n < 0) {
    return absl::InternalError(
        tensorstore::StrCat("Internal blosc error: ", n));
  }
  return n;
}

Result<std::string> Decode(std::string_view input) {
  std::string output;
  auto result = DecodeWithCallback(input, [&](size_t n) {
    output.resize(n);
    return output.data();
  });
  if (!result.ok()) return result.status();
  return output;
}

Result<size_t> DecodeWithCallback(
    std::string_view input,
    absl::FunctionRef<char*(size_t)> get_output_buffer) {
  TENSORSTORE_ASSIGN_OR_RETURN(size_t nbytes, GetDecodedSize(input));
  char* output_buffer = get_output_buffer(nbytes);
  if (!output_buffer) return 0;
  if (nbytes > 0) {
    const int n = blosc_decompress_ctx(input.data(), output_buffer, nbytes,
                                       /*numinternalthreads=*/1);
    if (n <= 0) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Blosc error: ", n));
    }
  }
  return nbytes;
}

Result<size_t> GetDecodedSize(std::string_view input) {
  size_t nbytes;
  if (blosc_cbuffer_validate(input.data(), input.size(), &nbytes) != 0) {
    return absl::InvalidArgumentError("Invalid blosc-compressed data");
  }
  return nbytes;
}

BloscWriter::BloscWriter(const blosc::Options& options,
                         riegeli::Writer& base_writer)
    : CordWriter(riegeli::CordWriterBase::Options().set_max_block_size(
          std::numeric_limits<size_t>::max())),
      options_(options),
      base_writer_(base_writer) {}

void BloscWriter::Done() {
  CordWriter::Done();
  auto result = blosc::EncodeWithCallback(dest().Flatten(), options_,
                                          [&](size_t n) -> char* {
                                            if (!base_writer_.Push(n)) {
                                              Fail(base_writer_.status());
                                              return nullptr;
                                            }
                                            return base_writer_.cursor();
                                          });
  if (!result.ok()) {
    Fail(std::move(result).status());
    return;
  }
  if (!*result) {
    // Already failed when encoding.
    return;
  }
  base_writer_.move_cursor(*result);
  if (!base_writer_.Close()) {
    Fail(base_writer_.status());
    return;
  }
}

BloscReader::BloscReader(riegeli::Reader& base_reader)
    : base_reader_(base_reader) {
  if (auto status = riegeli::ReadAll(base_reader_, encoded_data_);
      !status.ok()) {
    Fail(std::move(status));
    return;
  }
  if (auto result = blosc::GetDecodedSize(encoded_data_); result.ok()) {
    decoded_size_ = *result;
  } else {
    Fail(std::move(result).status());
  }
}

bool BloscReader::ToleratesReadingAhead() { return true; }
bool BloscReader::SupportsSize() { return true; }
bool BloscReader::PullSlow(size_t min_length, size_t recommended_length) {
  if (decoded_size_ == 0 || start() != nullptr || pos() > 0) {
    // Data was already decoded.  The precondition `min_length > available()`
    // for this method implies that `min_length` would exceed EOF.
    return false;
  }
  auto result = blosc::DecodeWithCallback(encoded_data_, [&](size_t n) {
    assert(n == decoded_size_);
    auto* buffer = new char[n];
    buffer_.reset(buffer);
    set_buffer(buffer, n);
    move_limit_pos(n);
    return buffer;
  });
  if (!result.ok()) {
    Fail(std::move(result).status());
    return false;
  }
  return min_length <= decoded_size_;
}

bool BloscReader::ReadSlow(size_t length, char* dest) {
  if (decoded_size_ == 0 || start() != nullptr || pos() > 0 ||
      length < decoded_size_) {
    // Use default implementation which may call `PullSlow`.
    return Reader::ReadSlow(length, dest);
  }
  if (auto result = blosc::DecodeWithCallback(encoded_data_,
                                              [&](size_t n) { return dest; });
      !result.ok()) {
    Fail(std::move(result).status());
    return false;
  }
  move_limit_pos(decoded_size_);
  return length == decoded_size_;
}

}  // namespace blosc
}  // namespace tensorstore
