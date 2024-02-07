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

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <variant>

#include "absl/base/internal/endian.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/digests/crc32c_digester.h"
#include "riegeli/digests/digesting_reader.h"
#include "riegeli/digests/digesting_writer.h"
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
  if (!reader.Pull()) {
    // Error status already set, or EOF.
    return false;
  }
  reader.Fail(absl::DataLossError("Invalid 64-bit varint value"));
  return false;
}

bool ReadVarintChecked(riegeli::Reader& reader, uint32_t& value) {
  if (riegeli::ReadVarint32(reader, value)) return true;
  if (!reader.Pull()) {
    // Error status already set, or EOF.
    return false;
  }
  reader.Fail(absl::DataLossError("Invalid 32-bit varint value"));
  return false;
}

bool ReadVarintChecked(riegeli::Reader& reader, uint16_t& value) {
  uint32_t temp;
  if (!ReadVarintChecked(reader, temp)) {
    // Error status already set, or EOF.
    return false;
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
  constexpr size_t kMinLength = 4     // magic
                                + 8   // length
                                + 4;  // crc32.

  if (encoded.size() < kMinLength) {
    return absl::DataLossError(
        absl::StrFormat("Encoded length (%d) is less than minimum length (%d)",
                        encoded.size(), kMinLength));
  }

  riegeli::CordReader reader(&encoded);

  // Reserve the final 4 bytes for the crc32
  riegeli::DigestingReader digesting_reader(
      riegeli::LimitingReader(
          &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                       encoded.size() - 4)),
      riegeli::Crc32cDigester());

  bool success = [&] {
    {
      uint32_t magic;
      if (!riegeli::ReadBigEndian<uint32_t>(digesting_reader, magic)) {
        return false;
      }
      if (magic != expected_magic) {
        digesting_reader.Fail(absl::DataLossError(absl::StrFormat(
            "Expected to start with hex bytes %08x but received: 0x%08x",
            expected_magic, magic)));
        return false;
      }

      uint64_t length;
      if (!riegeli::ReadLittleEndian<uint64_t>(digesting_reader, length)) {
        return false;
      }
      if (length != encoded.size()) {
        digesting_reader.Fail(absl::DataLossError(absl::StrFormat(
            "Length in header (%d) does not match actual length (%d)", length,
            encoded.size())));
        return false;
      }
    }

    uint32_t version;
    if (!ReadVarintChecked(digesting_reader, version)) return false;
    if (version > max_version_number) {
      digesting_reader.Fail(absl::DataLossError(
          absl::StrFormat("Maximum supported version is %d but received: %d",
                          max_version_number, version)));
      return false;
    }

    uint32_t compression_format;
    if (!ReadVarintChecked(digesting_reader, compression_format)) return false;

    bool success;
    switch (compression_format) {
      case 0:
        // Uncompressed
        success = decode_decompressed(digesting_reader, version);
        break;
      case 1: {
        riegeli::ZstdReader zstd_reader(&digesting_reader);
        success = decode_decompressed(zstd_reader, version) &&
                  zstd_reader.VerifyEndAndClose();
        if (!success && !zstd_reader.ok()) {
          digesting_reader.Fail(zstd_reader.status());
        }
        break;
      }
      default:
        digesting_reader.Fail(absl::DataLossError(absl::StrFormat(
            "Unsupported compression format: %d", compression_format)));
        return false;
    }

    return success;
  }();
  TENSORSTORE_RETURN_IF_ERROR(
      internal_ocdbt::FinalizeReader(digesting_reader, success));

  uint32_t expected_digest;
  // Length was already checked previously.
  ABSL_CHECK(riegeli::ReadLittleEndian<uint32_t>(reader, expected_digest));

  if (digesting_reader.Digest() != expected_digest) {
    return absl::DataLossError(absl::StrFormat(
        "CRC-32C checksum verification failed: expected=%d, actual=%d",
        expected_digest, digesting_reader.Digest()));
  }
  return absl::OkStatus();
}

Result<absl::Cord> EncodeWithOptionalCompression(
    const Config& config, uint32_t magic, uint32_t version_number,
    absl::FunctionRef<bool(riegeli::Writer& writer)> encode) {
  absl::Cord encoded;
  riegeli::CordWriter writer(&encoded);
  bool success = [&] {
    char header[12];
    absl::big_endian::Store32(header, magic);

    // Leave 12-byte placeholder to be filled in later.
    if (!writer.WriteZeros(12)) return false;

    // Compute CRC-32C digest of remaining data.
    riegeli::DigestingWriter digesting_writer(&writer,
                                              riegeli::Crc32cDigester());
    if (!riegeli::WriteVarint32(version_number, digesting_writer)) return false;
    if (std::holds_alternative<Config::NoCompression>(config.compression)) {
      if (!riegeli::WriteVarint32(0, digesting_writer)) return false;
      if (!encode(digesting_writer)) return false;
    } else {
      if (!riegeli::WriteVarint32(1, digesting_writer)) return false;
      const auto& zstd_config =
          std::get<Config::ZstdCompression>(config.compression);
      riegeli::ZstdWriter zstd_writer(
          &digesting_writer,
          riegeli::ZstdWriterBase::Options().set_compression_level(
              zstd_config.level));
      if (!encode(zstd_writer) || !zstd_writer.Close()) {
        digesting_writer.Fail(zstd_writer.status());
        return false;
      }
    }
    if (!digesting_writer.Close()) {
      writer.Fail(digesting_writer.status());
    }

    // Complete `header` by filling in length.
    auto length = writer.pos() + 4;
    absl::little_endian::Store64(header + 4, length);

    std::string_view header_string_view(header, sizeof(header));

    // Compute overall digest of header plus the remaining data.
    uint32_t digest = digesting_writer.Digest();
    digest = static_cast<uint32_t>(absl::ConcatCrc32c(
        absl::ComputeCrc32c(header_string_view), absl::crc32c_t(digest),
        length - header_string_view.size() - 4));

    // Append the digest.
    if (!riegeli::WriteLittleEndian<uint32_t>(digest, writer)) return false;

    // Replace the placeholder zero bytes with the header.
    if (!writer.Seek(0)) return false;
    if (!writer.Write(header_string_view)) return false;

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

size_t FindCommonPrefixLength(std::string_view a, std::string_view b) {
  size_t max_length = std::min(a.size(), b.size());
  for (size_t prefix_length = 0; prefix_length < max_length; ++prefix_length) {
    if (a[prefix_length] != b[prefix_length]) return prefix_length;
  }
  return max_length;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
