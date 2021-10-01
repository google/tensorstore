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

#ifndef TENSORSTORE_SERIALIZATION_RIEGELI_DELIMITED_H_
#define TENSORSTORE_SERIALIZATION_RIEGELI_DELIMITED_H_

#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/varint/varint_reading.h"
#include "riegeli/varint/varint_writing.h"

namespace tensorstore {
namespace serialization {

/// Writes a `size_t` value in varint format.
[[nodiscard]] inline bool WriteSize(riegeli::Writer& writer, size_t size) {
  if constexpr (sizeof(size_t) == 4) {
    return riegeli::WriteVarint32(size, writer);
  } else {
    return riegeli::WriteVarint64(size, writer);
  }
}

namespace internal_serialization {
void FailInvalidSize(riegeli::Reader& reader);
}  // namespace internal_serialization

/// Reads a `size_t` value as written by `WriteSize`.
[[nodiscard]] inline bool ReadSize(riegeli::Reader& reader, size_t& dest) {
  if constexpr (sizeof(size_t) == 4) {
    uint32_t value;
    if (!riegeli::ReadVarint32(reader, value)) {
      internal_serialization::FailInvalidSize(reader);
      return false;
    }
    dest = value;
    return true;
  } else {
    uint64_t value;
    if (!riegeli::ReadVarint64(reader, value)) {
      internal_serialization::FailInvalidSize(reader);
      return false;
    }
    dest = value;
    return true;
  }
}

/// Writes a length-delimited string.
///
/// \tparam Src May be any string type supported by `riegeli::Writer::Write`,
///     including `std::string_view`, `std::string`, `absl::Cord`, and
///     `riegeli::Chain`.
template <typename Src>
[[nodiscard]] inline bool WriteDelimited(riegeli::Writer& writer, Src&& src) {
  return WriteSize(writer, src.size()) && writer.Write(std::forward<Src>(src));
}

/// Reads a length-delimited string as written by `ReadDelimited`.
///
/// Sets `dest` to refer to the string.  Any use of `reader` will invalidate
/// `dest`.
[[nodiscard]] inline bool ReadDelimited(riegeli::Reader& reader,
                                        std::string_view& dest) {
  size_t size;
  return serialization::ReadSize(reader, size) && reader.Read(size, dest);
}

/// Reads a length-delimited string as written by `ReadDelimited`, and assigns
/// the result to `dest`.
[[nodiscard]] inline bool ReadDelimited(riegeli::Reader& reader,
                                        std::string& dest) {
  size_t size;
  return serialization::ReadSize(reader, size) && reader.Read(size, dest);
}

/// Reads a length-delimited string as written by `ReadDelimited`, and assigns
/// the result to `dest`.
[[nodiscard]] inline bool ReadDelimited(riegeli::Reader& reader,
                                        absl::Cord& dest) {
  size_t size;
  return serialization::ReadSize(reader, size) && reader.Read(size, dest);
}

/// Reads a length-delimited string as written by `WriteDelimited` and validates
/// that it is UTF-8.
bool ReadDelimitedUtf8(riegeli::Reader& reader, std::string& dest);

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_RIEGELI_DELIMITED_H_
