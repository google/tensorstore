// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/compression/zlib_compressor.h"

#include <stddef.h>

#include <memory>
#include <utility>

#include "riegeli/bytes/writer.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"

namespace tensorstore {
namespace internal {

std::unique_ptr<riegeli::Writer> ZlibCompressor::GetWriter(
    riegeli::Writer& base_writer, size_t element_bytes) const {
  using Writer = riegeli::ZlibWriter<riegeli::Writer*>;
  Writer::Options options;
  if (level != -1) options.set_compression_level(level);
  options.set_header(use_gzip_header ? Writer::Header::kGzip
                                     : Writer::Header::kZlib);
  return std::make_unique<Writer>(&base_writer, options);
}

std::unique_ptr<riegeli::Reader> ZlibCompressor::GetReader(
    riegeli::Reader& base_reader, size_t element_bytes) const {
  using Reader = riegeli::ZlibReader<riegeli::Reader*>;
  Reader::Options options;
  options.set_header(use_gzip_header ? Reader::Header::kGzip
                                     : Reader::Header::kZlib);
  return std::make_unique<Reader>(&base_reader, options);
}

}  // namespace internal
}  // namespace tensorstore
