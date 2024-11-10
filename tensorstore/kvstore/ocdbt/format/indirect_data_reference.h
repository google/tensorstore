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

#include <stdint.h>

#include <iosfwd>
#include <limits>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Specifies the kind of data referenced by an `IndirectDataReference`.
enum class IndirectDataKind {
  /// Indirect value, i.e. value not stored inline a leaf B+tree node.
  kValue,
  /// Encoded B+tree node.
  kBtreeNode,
  /// Encoded version tree node.
  kVersionNode,
};

constexpr size_t kNumIndirectDataKinds = 3;

Result<IndirectDataKind> ParseIndirectDataKind(std::string_view label);
std::string_view IndirectDataKindToString(IndirectDataKind kind);

std::ostream& operator<<(std::ostream& os, IndirectDataKind kind);

/// In-memory representation of a given byte range within a given data file.
struct IndirectDataReference {
  DataFileId file_id;
  uint64_t offset;
  uint64_t length;

  /// Checks that the offset/length pair is valid.
  absl::Status Validate(bool allow_missing) const;

  /// Encodes as a string key.
  friend void EncodeCacheKeyAdl(std::string* out,
                                const IndirectDataReference& self);
  std::string EncodeCacheKey() const {
    std::string out;
    EncodeCacheKeyAdl(&out, *this);
    return out;
  }

  /// Decodes from the result of `EncodeCacheKey`.
  bool DecodeCacheKey(std::string_view encoded);

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
