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

#include "tensorstore/internal/compression/zstd_compressor.h"

#include <cstddef>

#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"

namespace tensorstore {
namespace internal {

std::unique_ptr<riegeli::Writer> ZstdCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  using Writer = riegeli::ZstdWriter<std::unique_ptr<riegeli::Writer>>;
  Writer::Options options;
  options.set_compression_level(level);
  return std::make_unique<Writer>(std::move(base_writer), options);
}

std::unique_ptr<riegeli::Reader> ZstdCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  using Reader = riegeli::ZstdReader<std::unique_ptr<riegeli::Reader>>;
  return std::make_unique<Reader>(std::move(base_reader));
}

}  // namespace internal
}  // namespace tensorstore
