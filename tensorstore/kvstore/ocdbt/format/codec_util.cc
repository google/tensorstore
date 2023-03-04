// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/codec_util.h"

#include <cstdint>
#include <limits>
#include <string>
#include <variant>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "riegeli/varint/varint_reading.h"
#include "riegeli/varint/varint_writing.h"
#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {

bool ReadVarintChecked(riegeli::Reader& reader, uint64_t& value) {
  if (riegeli::ReadVarint64(reader, value)) return true;
  if (!reader.ok() || !reader.available()) {
    // Error status already set, or EOF.
    return false;
  }
  reader.Fail(absl::DataLossError("Invalid 64-bit varint value"));
  return false;
}

bool ReadVarintChecked(riegeli::Reader& reader, uint32_t& value) {
  if (riegeli::ReadVarint32(reader, value)) return true;
  if (!reader.ok() || !reader.available()) {
    // Error status already set, or EOF.
    return false;
  }
  reader.Fail(absl::DataLossError("Invalid 32-bit varint value"));
  return false;
}

bool ReadVarintChecked(riegeli::Reader& reader, uint16_t& value) {
  uint32_t temp;
  if (!riegeli::ReadVarint32(reader, temp)) {
    if (!reader.ok() || !reader.available()) {
      // Error status already set, or EOF.
      return false;
    }
  }
  if (temp <= std::numeric_limits<uint16_t>::max()) {
    value = static_cast<uint16_t>(temp);
    return true;
  }
  reader.Fail(absl::DataLossError("Invalid 16-bit varint value"));
  return false;
}

absl::Status DecodeWithOptionalCompression(
    const absl::Cord& encoded, uint32_t expected_magic,
    uint32_t max_version_number,
    absl::FunctionRef<bool(riegeli::Reader& reader, uint32_t version)>
        decode_decompressed) {
  riegeli::CordReader reader{encoded};
  bool success = [&] {
    {
      uint32_t magic;
      if (!riegeli::ReadBigEndian<uint32_t>(reader, magic)) {
        return false;
      }
      if (magic != expected_magic) {
        reader.Fail(absl::DataLossError(absl::StrFormat(
            "Expected to start with hex bytes %08x but received: 0x%08x",
            expected_magic, magic)));
        return false;
      }
    }

    uint32_t version;
    if (!ReadVarintChecked(reader, version)) return false;
    if (version > max_version_number) {
      reader.Fail(absl::DataLossError(
          absl::StrFormat("Maximum supported version is %d but received: %d",
                          max_version_number, version)));
      return false;
    }

    uint32_t compression_format;
    if (!ReadVarintChecked(reader, compression_format)) return false;

    bool success;
    switch (compression_format) {
      case 0:
        // Uncompressed
        success = decode_decompressed(reader, version);
        break;
      case 1: {
        riegeli::ZstdReader<> zstd_reader(&reader);
        success = decode_decompressed(zstd_reader, version) &&
                  zstd_reader.VerifyEndAndClose();
        if (!success && !zstd_reader.ok()) {
          reader.Fail(zstd_reader.status());
        }
        break;
      }
      default:
        reader.Fail(absl::DataLossError(absl::StrFormat(
            "Unsupported compression format: %d", compression_format)));
        return false;
    }
    return success;
  }();
  return internal_ocdbt::FinalizeReader(reader, success);
}

Result<absl::Cord> EncodeWithOptionalCompression(
    const Config& config, uint32_t magic, uint32_t version_number,
    absl::FunctionRef<bool(riegeli::Writer& writer)> encode) {
  absl::Cord encoded;
  riegeli::CordWriter<> writer{&encoded};
  bool success = [&] {
    if (!riegeli::WriteBigEndian<uint32_t>(magic, writer)) return false;
    if (!riegeli::WriteVarint32(version_number, writer)) return false;
    if (std::holds_alternative<Config::NoCompression>(config.compression)) {
      if (!riegeli::WriteVarint32(0, writer)) return false;
      return encode(writer);
    }
    if (!riegeli::WriteVarint32(1, writer)) return false;
    const auto& zstd_config =
        std::get<Config::ZstdCompression>(config.compression);
    riegeli::ZstdWriter<> zstd_writer{
        &writer, riegeli::ZstdWriter<>::Options().set_compression_level(
                     zstd_config.level)};
    if (!encode(zstd_writer) || !zstd_writer.Close()) {
      writer.Fail(zstd_writer.status());
      return false;
    }
    return true;
  }();
  TENSORSTORE_RETURN_IF_ERROR(internal_ocdbt::FinalizeWriter(writer, success));
  return encoded;
}

absl::Status FinalizeReader(riegeli::Reader& reader, bool success) {
  if (!success && reader.ok()) {
    reader.Fail(absl::DataLossError("Unexpected end of data"));
  }
  if (success && reader.VerifyEndAndClose()) return absl::OkStatus();
  return reader.status();
}

absl::Status FinalizeWriter(riegeli::Writer& writer, bool success) {
  if (success && writer.Close()) {
    return absl::OkStatus();
  }
  return writer.status();
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
