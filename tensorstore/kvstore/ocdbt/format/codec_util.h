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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_CODEC_UTIL_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_CODEC_UTIL_H_

/// \file
///
/// Common utilities used to encode/decode data structures used by the database.
///
/// Some data structures and encoded and decoded using instances of the
/// `Codec<T>` concept.  A type satisfies the `Codec<T>` concept if it is a
/// function object with the signatures:
///
///   bool (riegeli::Reader &reader, T& value);
///   bool (riegeli::Writer &writer, const T& value);

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Reads a varint.
///
/// Unlike `riegeli::ReadVarint{64,32}`, in the case of an error, marks `reader`
/// as failed.
///
/// \returns `true` on success, `false` on error.
bool ReadVarintChecked(riegeli::Reader& reader, uint64_t& value);
bool ReadVarintChecked(riegeli::Reader& reader, uint32_t& value);
bool ReadVarintChecked(riegeli::Reader& reader, uint16_t& value);

/// Writes a varint.
///
/// \tparam T Must be one of `uint64_t`, `uint32_t`, or `uint16_t`.
///
/// \returns `true` on success, `false` on error.
template <typename T>
bool WriteVarint(riegeli::Writer& writer, T value);

template <>
inline bool WriteVarint<uint64_t>(riegeli::Writer& writer, uint64_t value) {
  return riegeli::WriteVarint64(value, writer);
}

template <>
inline bool WriteVarint<uint32_t>(riegeli::Writer& writer, uint32_t value) {
  return riegeli::WriteVarint32(value, writer);
}

template <>
inline bool WriteVarint<uint16_t>(riegeli::Writer& writer, uint16_t value) {
  return riegeli::WriteVarint32(value, writer);
}

/// Codec for varint values.
template <typename T>
struct VarintCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader, T& value) const {
    return ReadVarintChecked(reader, value);
  }

  [[nodiscard]] bool operator()(riegeli::Writer& writer, T value) const {
    return WriteVarint<T>(writer, value);
  }
};

/// Codec for fixed-size little-endian numeric values.
template <typename T>
struct LittleEndianCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader, T& value) const {
    return riegeli::ReadLittleEndian<T>(reader, value);
  }

  [[nodiscard]] bool operator()(riegeli::Writer& writer, T value) const {
    return riegeli::WriteLittleEndian<T>(value, writer);
  }
};

template <typename T>
struct RawBytesCodec {
  [[nodiscard]] bool operator()(riegeli::Reader& reader, T& value) const {
    return reader.Read(sizeof(T), reinterpret_cast<char*>(&value));
  }

  [[nodiscard]] bool operator()(riegeli::Writer& writer, const T& value) const {
    return writer.Write(
        absl::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
  }
};

struct NoOpCodec {
  template <typename IO, typename T>
  bool operator()(IO& io, T&& value) const {
    return true;
  }
};

// TODO(jbms): Add checksum to the common compression header.

// TODO(jbms): Consider changing `{Decode,Encode}WithOptionalCompression` to use
// class interface rather than callback interface.

/// Decodes the common compression header.
///
/// \param encoded The encoded representation.
/// \param expected_magic Expected magic number at start of header.
/// \param max_version_number Maximum allowed version number in header.
/// \param decode_compressed Callback to be invoked to encode the uncompressed
///     body.
absl::Status DecodeWithOptionalCompression(
    const absl::Cord& encoded, uint32_t expected_magic,
    uint32_t max_version_number,
    absl::FunctionRef<bool(riegeli::Reader& reader, uint32_t version)>
        decode_decompressed);

/// Encodes with the common compression header.
///
/// \param config Configuration specify compression options.
/// \param magic Magic number to include at start of header.
/// \param version_number Version number to include in header.
/// \param encode Callback to be invoked to encode the uncompressed body.
Result<absl::Cord> EncodeWithOptionalCompression(
    const Config& config, uint32_t magic, uint32_t version_number,
    absl::FunctionRef<bool(riegeli::Writer& writer)> encode);

/// Closes `reader`, verifying that the end has been reached and
/// `success == true`.
absl::Status FinalizeReader(riegeli::Reader& reader, bool success);

/// Closes `writer` if `success == true`.
absl::Status FinalizeWriter(riegeli::Writer& writer, bool success);

/// Returns the longest common prefix of `a` and `b`.
size_t FindCommonPrefixLength(std::string_view a, std::string_view b);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_CODEC_UTIL_H_
