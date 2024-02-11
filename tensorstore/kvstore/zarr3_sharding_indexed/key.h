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

#ifndef TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_KEY_H_
#define TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_KEY_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <string_view>

#include "tensorstore/index.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

/// Identifies an entry within a shard.
///
/// For a given grid cell specified by an `indices` vector, the entry id is
/// equal to ``GetContiguousOffset<c_order>(grid_shape, indices)``.
using EntryId = uint32_t;

/// Converts an index vector to the key representation.
///
/// This just encodes each index as a uint32be value, for a total length of
/// `grid_cell_indices.size() * 4`.
std::string IndicesToKey(span<const Index> grid_cell_indices);

/// Converts a key to an index vector.
///
/// This is the inverse of `IndicesToKey`.
bool KeyToIndices(std::string_view key, span<Index> grid_cell_indices);

/// Converts a string key to an entry id.
///
/// For a given `indices` vector, the string key of size `indices.size() * 4` is
/// obtained by concatenating (in order) the uint32be encoding of each index.
std::optional<EntryId> KeyToEntryId(std::string_view key,
                                    span<const Index> grid_shape);
Result<EntryId> KeyToEntryIdOrError(std::string_view key,
                                    span<const Index> grid_shape);

/// Converts an entry id to a string key.
///
/// This is the inverse of `KeyToEntryId`.
std::string EntryIdToKey(EntryId entry_id, span<const Index> grid_shape);

/// Computes the first `EntryId` whose string key representation is not
/// lexicographically less than `key`.
EntryId LowerBoundToEntryId(std::string_view key, span<const Index> grid_shape);

/// Computes the range of entry ids whose string key representations are
/// contained within the key range given by `inclusive_min` and `exclusive_max`.
std::pair<EntryId, EntryId> KeyRangeToEntryRange(std::string_view inclusive_min,
                                                 std::string_view exclusive_max,
                                                 span<const Index> grid_shape);

/// Converts an `entry_id` to the internal key representation.
///
/// The internal key representation is the 4-byte uint32be representation of
/// `entry_id`.  It is used in internal data structures in place of the external
/// string key representation for efficiency.
std::string EntryIdToInternalKey(EntryId entry_id);

/// Converts from the internal key representation to an `EntryId`.
EntryId InternalKeyToEntryId(std::string_view key);

/// Same as `LowerBoundToEntryId`, but operates on the internal key
/// representation.
EntryId InternalKeyLowerBoundToEntryId(std::string_view key,
                                       int64_t num_entries_per_shard);

/// Same as `KeyRangeToEntryRange`, but operates on the internal key
/// representation.
std::pair<EntryId, EntryId> InternalKeyRangeToEntryRange(
    std::string_view inclusive_min, std::string_view exclusive_max,
    int64_t num_entries_per_shard);

/// Applies `KeyRangeToEntryRange`, then converts each entry back to an internal
/// key.
KeyRange KeyRangeToInternalKeyRange(const KeyRange& range,
                                    span<const Index> grid_shape);

/// Returns a human-readable description of the key, for use in error messages.
std::string DescribeEntryId(EntryId entry_id, span<const Index> grid_shape);
std::string DescribeKey(std::string_view key, span<const Index> grid_shape);
std::string DescribeInternalKey(std::string_view key,
                                span<const Index> grid_shape);

}  // namespace zarr3_sharding_indexed
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_KEY_H_
