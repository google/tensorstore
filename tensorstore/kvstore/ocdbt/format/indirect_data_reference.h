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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_H_

#include <array>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <string>
#include <string_view>

#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Identifies a data file by a 128-bit (random) identifier.
///
/// 128 bits is sufficient to ensure that random ids can be generated without a
/// need to check for collisions.
///
/// The corresponding key in the underlying kvstore is obtained from
/// `GetDataFilePath`.
struct DataFileId {
  using Value = std::array<uint8_t, 16>;
  Value value;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.value);
  };

  friend std::ostream& operator<<(std::ostream& os, const DataFileId& x);

  friend bool operator==(const DataFileId& a, const DataFileId& b) {
    return a.value == b.value;
  }
  friend bool operator!=(const DataFileId& a, const DataFileId& b) {
    return !(a == b);
  }
};

/// Generates a random data file id.
DataFileId GenerateDataFileId();

/// Returns the kvstore key for a given file identifier.
///
/// This key is relative to the data directory path returned by
/// `GetDataDirectoryPath`.
///
/// Equal to the 32-character lowercase hex representation of `file_id`.
std::string GetDataFilePath(DataFileId file_id);

/// Returns the path to the data directory in the underlying kvstore given a
/// base directory path.
///
/// Equal to `base_path + "d/"`.
std::string GetDataDirectoryPath(std::string_view base_path);

/// In-memory representation of a given byte range within a given data file.
struct IndirectDataReference {
  DataFileId file_id;
  uint64_t offset;
  uint64_t length;

  /// Returns the special value that indicates an invalid/null reference.
  ///
  /// This is used in only one instance: when referring to the root node of a
  /// version, to indicate an empty b+tree.
  static IndirectDataReference Missing() {
    IndirectDataReference ref = {};
    ref.offset = std::numeric_limits<uint64_t>::max();
    ref.length = std::numeric_limits<uint64_t>::max();
    return ref;
  }

  /// Returns `true` if this is equal to `Missing()`.
  bool IsMissing() const {
    return offset == std::numeric_limits<uint64_t>::max() &&
           length == std::numeric_limits<uint64_t>::max();
  }

  /// Compares two references for equality.
  friend bool operator==(const IndirectDataReference& a,
                         const IndirectDataReference& b);
  friend bool operator!=(const IndirectDataReference& a,
                         const IndirectDataReference& b) {
    return !(a == b);
  }

  /// Prints a debugging representation.
  friend std::ostream& operator<<(std::ostream& os,
                                  const IndirectDataReference& x);
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.file_id, x.offset, x.length);
  };
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_INDIRECT_DATA_REFERENCE_H_
