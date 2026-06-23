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

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "absl/log/absl_check.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"

namespace tensorstore {
namespace internal {

std::unique_ptr<riegeli::Writer> ZstdCompressor::GetWriter(
    riegeli::Writer& base_writer, size_t element_bytes,
    int64_t pledged_size) const {
  using Writer = riegeli::ZstdWriter<riegeli::Writer*>;
  Writer::Options options;
  options.set_compression_level(level);
  ABSL_DCHECK_GE(pledged_size, -1);
  if (pledged_size >= 0) {
    options.set_pledged_size(static_cast<uint64_t>(pledged_size));
  }
  return std::make_unique<Writer>(&base_writer, options);
}

std::unique_ptr<riegeli::Reader> ZstdCompressor::GetReader(
    riegeli::Reader& base_reader, size_t element_bytes) const {
  using Reader = riegeli::ZstdReader<riegeli::Reader*>;
  return std::make_unique<Reader>(&base_reader);
}

}  // namespace internal
}  // namespace tensorstore
