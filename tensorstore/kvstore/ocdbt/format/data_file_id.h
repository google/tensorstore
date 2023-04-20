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

#ifndef TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_H_
#define TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_H_

#include <iosfwd>
#include <string>

#include "tensorstore/internal/ref_counted_string.h"

namespace tensorstore {
namespace internal_ocdbt {

using BasePath = internal::RefCountedString;
using RelativePath = internal::RefCountedString;

/// Identifies a data file by a `base_path` and `relative_path`.
///
/// The actual data file is located by concatenating the ``current_prefix``
/// (which defaults to the OCDBT database root prefix), the `base_path`, and the
/// `relative_path`.
///
/// If an OCDBT B+tree or version tree node is located using this `DataFileId`
/// (x), then any `DataFileId` (y) specified in the node is resolved by
/// concatenating ``current_prefix``, ``x.base_path``, ``y.base_path``,
/// ``y.relative_path``.  Note that ``x.relative_path`` is not included.  If
/// this `DataFileId` merely specifies the location of a raw out-of-line value,
/// then the separation between `base_path` and `relative_path` is arbitrary.
///
/// In the common case, `base_path` will be the empty string, and
/// `relative_path` specifies the path to the data file relative to the root
/// directory/prefix of the OCDBT database,
/// e.g. "d/0123456789abcdef0123456789abcdef".
///
/// A non-empty `base_path` may be used to reference nodes in a separate,
/// existing OCDBT database, in conjunction with either symbolic links or a
/// KvStore adapter that maps a sub-prefix of the OCDBT database root prefix to
/// a different location.
struct DataFileId {
  BasePath base_path;
  RelativePath relative_path;

  std::string FullPath() const;
  size_t size() const { return base_path.size() + relative_path.size(); }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base_path, x.relative_path);
  };

  friend std::ostream& operator<<(std::ostream& os, const DataFileId& x);

  friend bool operator==(const DataFileId& a, const DataFileId& b) {
    return a.base_path == b.base_path && a.relative_path == b.relative_path;
  }
  friend bool operator!=(const DataFileId& a, const DataFileId& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const DataFileId& id) {
    return H::combine(std::move(h), id.base_path, id.relative_path);
  }
};

/// Generates a random data file id, where the `base_path` is "" and the
/// `relative_path` is of the form "d/0123456789abcdef0123456789abcdef".
DataFileId GenerateDataFileId();

using PathLength = uint16_t;

constexpr size_t kMaxPathLength = 65535;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_FORMAT_DATA_FILE_ID_H_
