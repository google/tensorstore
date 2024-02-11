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

#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

std::string IndicesToKey(span<const Index> grid_cell_indices) {
  std::string key;
  key.resize(grid_cell_indices.size() * 4);
  for (DimensionIndex i = 0; i < grid_cell_indices.size(); ++i) {
    absl::big_endian::Store32(key.data() + i * 4, grid_cell_indices[i]);
  }
  return key;
}

bool KeyToIndices(std::string_view key, span<Index> grid_cell_indices) {
  if (key.size() != grid_cell_indices.size() * 4) {
    return false;
  }
  for (DimensionIndex i = 0; i < grid_cell_indices.size(); ++i) {
    grid_cell_indices[i] = absl::big_endian::Load32(key.data() + i * 4);
  }
  return true;
}

std::optional<EntryId> KeyToEntryId(std::string_view key,
                                    span<const Index> grid_shape) {
  const DimensionIndex rank = grid_shape.size();
  if (rank * sizeof(uint32_t) != key.size()) return {};
  EntryId id = 0;
  for (DimensionIndex i = 0; i < rank; ++i) {
    auto index = absl::big_endian::Load32(key.data() + i * 4);
    if (index >= grid_shape[i]) return {};
    id *= grid_shape[i];
    id += index;
  }
  return id;
}
Result<EntryId> KeyToEntryIdOrError(std::string_view key,
                                    span<const Index> grid_shape) {
  if (auto entry_id = KeyToEntryId(key, grid_shape)) {
    return *entry_id;
  }
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Invalid key (grid_shape=", grid_shape,
                          "): ", tensorstore::QuoteString(key)));
}

std::string EntryIdToKey(EntryId entry_id, span<const Index> grid_shape) {
  std::string key;
  key.resize(grid_shape.size() * 4);
  for (DimensionIndex i = grid_shape.size(); i--;) {
    const Index size = grid_shape[i];
    absl::big_endian::Store32(key.data() + i * 4, entry_id % size);
    entry_id /= size;
  }
  return key;
}

EntryId LowerBoundToEntryId(std::string_view key,
                            span<const Index> grid_shape) {
  char key_padded[kMaxRank * 4];
  const size_t full_key_size = grid_shape.size() * 4;
  const size_t key_bytes_to_copy = std::min(full_key_size, key.size());
  std::memcpy(key_padded, key.data(), key_bytes_to_copy);
  std::memset(key_padded + key_bytes_to_copy, 0,
              full_key_size - key_bytes_to_copy);
  EntryId entry_id = 0;
  EntryId remaining_indices_mask = ~static_cast<EntryId>(0);
  EntryId max_entry_id = 1;
  for (DimensionIndex i = 0; i < grid_shape.size(); ++i) {
    const EntryId size = grid_shape[i];
    max_entry_id *= size;
    EntryId index = absl::big_endian::Load32(&key_padded[i * 4]);
    entry_id *= size;
    if (index >= size) {
      entry_id += (size & remaining_indices_mask);
      remaining_indices_mask = 0;
    } else {
      entry_id += (index & remaining_indices_mask);
    }
  }
  assert(entry_id <= max_entry_id);
  if (key.size() > full_key_size) {
    if (entry_id < max_entry_id) {
      ++entry_id;
    }
  }
  return entry_id;
}

std::pair<EntryId, EntryId> KeyRangeToEntryRange(std::string_view inclusive_min,
                                                 std::string_view exclusive_max,
                                                 span<const Index> grid_shape) {
  EntryId lower_bound = LowerBoundToEntryId(inclusive_min, grid_shape);
  EntryId upper_bound;
  if (exclusive_max.empty()) {
    upper_bound = static_cast<EntryId>(ProductOfExtents(grid_shape));
  } else {
    upper_bound = LowerBoundToEntryId(exclusive_max, grid_shape);
  }
  return {lower_bound, upper_bound};
}

EntryId InternalKeyLowerBoundToEntryId(std::string_view key,
                                       int64_t num_entries_per_shard) {
  char key_bytes[4] = {};
  std::memcpy(key_bytes, key.data(),
              std::min(static_cast<size_t>(4), key.size()));
  EntryId entry_id = absl::big_endian::Load32(key_bytes);
  if (entry_id > num_entries_per_shard) {
    entry_id = num_entries_per_shard;
  }
  if (key.size() > 4 && entry_id < num_entries_per_shard) {
    ++entry_id;
  }
  return entry_id;
}

std::pair<EntryId, EntryId> InternalKeyRangeToEntryRange(
    std::string_view inclusive_min, std::string_view exclusive_max,
    int64_t num_entries_per_shard) {
  return {InternalKeyLowerBoundToEntryId(inclusive_min, num_entries_per_shard),
          exclusive_max.empty() ? EntryId(num_entries_per_shard)
                                : InternalKeyLowerBoundToEntryId(
                                      exclusive_max, num_entries_per_shard)};
}

std::string EntryIdToInternalKey(EntryId entry_id) {
  std::string key;
  key.resize(4);
  absl::big_endian::Store32(key.data(), entry_id);
  return key;
}

EntryId InternalKeyToEntryId(std::string_view key) {
  assert(key.size() == 4);
  return static_cast<EntryId>(absl::big_endian::Load32(key.data()));
}

KeyRange KeyRangeToInternalKeyRange(const KeyRange& range,
                                    span<const Index> grid_shape) {
  auto [inclusive_min_entry, exclusive_max_entry] = KeyRangeToEntryRange(
      range.inclusive_min, range.exclusive_max, grid_shape);
  return KeyRange{EntryIdToInternalKey(inclusive_min_entry),
                  EntryIdToInternalKey(exclusive_max_entry)};
}

std::string DescribeEntryId(EntryId entry_id, span<const Index> grid_shape) {
  Index indices[kMaxRank];
  span<Index> indices_span(&indices[0], grid_shape.size());
  GetContiguousIndices<c_order, Index>(entry_id, grid_shape, indices_span);
  return tensorstore::StrCat("shard entry ", indices_span, "/", grid_shape);
}

std::string DescribeKey(std::string_view key, span<const Index> grid_shape) {
  if (auto entry_id = KeyToEntryId(key, grid_shape)) {
    return DescribeEntryId(*entry_id, grid_shape);
  }
  return tensorstore::StrCat("invalid shard entry ",
                             tensorstore::QuoteString(key), "/", grid_shape);
}

std::string DescribeInternalKey(std::string_view key,
                                span<const Index> grid_shape) {
  return DescribeEntryId(InternalKeyToEntryId(key), grid_shape);
}

}  // namespace zarr3_sharding_indexed
}  // namespace tensorstore
