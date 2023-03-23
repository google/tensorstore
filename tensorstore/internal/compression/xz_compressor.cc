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

#include "tensorstore/internal/compression/xz_compressor.h"

#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/xz/xz_reader.h"
#include "riegeli/xz/xz_writer.h"

namespace tensorstore {
namespace internal {

std::unique_ptr<riegeli::Writer> XzCompressor::GetWriter(
    std::unique_ptr<riegeli::Writer> base_writer, size_t element_bytes) const {
  using Writer = riegeli::XzWriter<std::unique_ptr<riegeli::Writer>>;
  Writer::Options options;
  options.set_container(Writer::Container::kXz);
  options.set_check(static_cast<Writer::Check>(check));
  options.set_compression_level(level);
  options.set_extreme(extreme);
  return std::make_unique<Writer>(std::move(base_writer), options);
}

std::unique_ptr<riegeli::Reader> XzCompressor::GetReader(
    std::unique_ptr<riegeli::Reader> base_reader, size_t element_bytes) const {
  using Reader = riegeli::XzReader<std::unique_ptr<riegeli::Reader>>;
  Reader::Options options;
  options.set_container(Reader::Container::kXzOrLzma);
  options.set_concatenate(true);
  return std::make_unique<Reader>(std::move(base_reader), options);
}

}  // namespace internal
}  // namespace tensorstore
