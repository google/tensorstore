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

#include "tensorstore/kvstore/ocdbt/format/config_codec.h"

#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {
struct ZstdCompressionOptionsCodec {
  template <typename IO, typename T>
  [[nodiscard]] bool operator()(IO& io, T&& value) const {
    static_assert(std::is_same_v<IO, riegeli::Reader> ||
                  std::is_same_v<IO, riegeli::Writer>);
    if (!LittleEndianCodec<int32_t>{}(io, value.level)) return false;
    if constexpr (std::is_same_v<IO, riegeli::Reader>) {
      using Options = riegeli::ZstdWriterBase::Options;
      if (value.level < Options::kMinCompressionLevel ||
          value.level > Options::kMaxCompressionLevel) {
        io.Fail(absl::InvalidArgumentError(absl::StrFormat(
            "Zstd compression level %d is outside valid range [%d, %d]",
            value.level, Options::kMinCompressionLevel,
            Options::kMaxCompressionLevel)));
      }
    }
    return true;
  }
};

using CompressionMethodCodec = VarintCodec<uint32_t>;
}  // namespace

bool CompressionConfigCodec::operator()(riegeli::Reader& reader,
                                        Config::Compression& value) const {
  uint32_t compression_method;
  if (!CompressionMethodCodec{}(reader, compression_method)) return false;
  switch (compression_method) {
    case 0:
      value.emplace<Config::NoCompression>();
      break;
    case 1:
      if (!ZstdCompressionOptionsCodec{}(
              reader, value.emplace<Config::ZstdCompression>())) {
        return false;
      }
      break;
    default:
      reader.Fail(absl::InvalidArgumentError(absl::StrFormat(
          "Invalid compression method: %d", compression_method)));
      return false;
  }
  return true;
}

bool CompressionConfigCodec::operator()(
    riegeli::Writer& writer, const Config::Compression& value) const {
  if (std::holds_alternative<Config::NoCompression>(value)) {
    if (!CompressionMethodCodec{}(writer, 0)) {
      return false;
    }
  } else {
    if (!CompressionMethodCodec{}(writer, 1) ||
        !ZstdCompressionOptionsCodec{}(
            writer, std::get<Config::ZstdCompression>(value))) {
      return false;
    }
  }
  return true;
}

bool ManifestKindCodec::operator()(riegeli::Reader& reader,
                                   ManifestKind& value) const {
  uint8_t manifest_kind;
  if (!reader.ReadByte(manifest_kind)) return false;
  if (manifest_kind > static_cast<size_t>(Config::kMaxManifestKind)) {
    reader.Fail(absl::DataLossError(
        absl::StrFormat("Invalid manifest_kind %d", manifest_kind)));
    return false;
  }
  value = static_cast<ManifestKind>(manifest_kind);
  return true;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
