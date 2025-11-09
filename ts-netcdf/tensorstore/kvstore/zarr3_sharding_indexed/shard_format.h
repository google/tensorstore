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

#ifndef TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_SHARD_FORMAT_H_
#define TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_SHARD_FORMAT_H_

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <limits>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/key.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

using internal_zarr3::ZarrCodecChain;
using internal_zarr3::ZarrCodecChainSpec;

/// Maximum supported size for a shard index.
constexpr int64_t kMaxNumEntries = 1024 * 1024 * 1024;

enum ShardIndexLocation {
  kStart,
  kEnd,
};

TENSORSTORE_DECLARE_JSON_BINDER(ShardIndexLocationJsonBinder,
                                ShardIndexLocation,
                                internal_json_binding::NoOptions,
                                internal_json_binding::NoOptions);

struct ShardIndexEntry {
  uint64_t offset = std::numeric_limits<uint64_t>::max();
  uint64_t length = std::numeric_limits<uint64_t>::max();

  static constexpr ShardIndexEntry Missing() { return ShardIndexEntry{}; }

  bool IsMissing() const {
    return offset == std::numeric_limits<uint64_t>::max() &&
           length == std::numeric_limits<uint64_t>::max();
  }

  /// Validates that the byte range is valid.
  ///
  /// The specified `entry_id` is used only for the error message.
  absl::Status Validate(EntryId entry_id) const;
  absl::Status Validate(EntryId entry_id, int64_t total_size) const;

  ByteRange AsByteRange() const {
    return ByteRange{static_cast<int64_t>(offset),
                     static_cast<int64_t>(offset + length)};
  }
};

/// Representation of decoded shard index.
struct ShardIndex {
  ShardIndexEntry operator[](int64_t i) const {
    assert(0 <= i &&
           i < ProductOfExtents(entries.shape().first(entries.rank() - 1)));
    return ShardIndexEntry{entries.data()[i * 2], entries.data()[i * 2 + 1]};
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.entries);
  };

  // Guaranteed to be in C order.  Inner dimension has a size of 2 and
  // corresponds to `[offset, length]`.  Outer dimensions correspond to the grid
  // shape.
  SharedArray<const uint64_t> entries;
};

/// Validates a grid shape.
absl::Status ValidateGridShape(span<const Index> grid_shape);

/// Initializes a codec chain spec for a given grid rank.
Result<ZarrCodecChain::Ptr> InitializeIndexCodecChain(
    const ZarrCodecChainSpec& codec_chain_spec, DimensionIndex grid_rank,
    ZarrCodecChainSpec* resolved_codec_chain_spec = nullptr);

/// Parameters used for encoding/decoding a shard index.
struct ShardIndexParameters {
  span<const Index> grid_shape() const {
    assert(index_shape.size() >= 0);
    return {index_shape.data(), static_cast<ptrdiff_t>(index_shape.size() - 1)};
  }

  // Initializes just `index_shape` and `num_entries`.
  absl::Status InitializeIndexShape(span<const Index> grid_shape);

  // Initializes all members.
  absl::Status Initialize(const ZarrCodecChain& codec_chain,
                          span<const Index> grid_shape);
  absl::Status Initialize(
      const ZarrCodecChainSpec& codec_chain_spec, span<const Index> grid_shape,
      ZarrCodecChainSpec* resolved_codec_chain_spec = nullptr);

  ShardIndexLocation index_location;

  // Equal to `ProductOfExtents(grid_shape())`.
  int64_t num_entries;

  // Equal to `grid_shape()` followed by `{2}`.
  std::vector<Index> index_shape;

  // Codec chain for decoding index.
  ZarrCodecChain::Ptr index_codec_chain;

  // Prepared state of `index_codec_chain` for `index_shape`.
  ZarrCodecChain::PreparedState::Ptr index_codec_state;
};

/// Decodes a shard index.
///
/// This does *not* validate the byte ranges.  Those must be validated before
/// use by calling `ShardIndexEntry::Validate`.
Result<ShardIndex> DecodeShardIndex(const absl::Cord& input,
                                    const ShardIndexParameters& parameters);

/// Decodes the shard index given the full shard.
///
/// This does *not* validate the byte ranges.  Those must be validated before
/// use by calling `ShardIndexEntry::Validate`.
Result<ShardIndex> DecodeShardIndexFromFullShard(
    const absl::Cord& shard_data,
    const ShardIndexParameters& shard_index_parameters);

/// Decoded representation of a single entry within a shard.
///
/// A missing key is indicated by `std::nullopt`.
using ShardEntry = std::optional<absl::Cord>;

/// Decoded representation of a complete shard.
struct ShardEntries {
  // Size must always match product of shard grid shape.
  std::vector<ShardEntry> entries;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.entries);
  };
};

/// Decodes a complete shard (entries followed by shard index).
Result<ShardEntries> DecodeShard(
    const absl::Cord& shard_data,
    const ShardIndexParameters& shard_index_parameters);

/// Encodes a complete shard (entries followed by shard index).
///
/// Returns `std::nullopt` if all entries are missing.
Result<std::optional<absl::Cord>> EncodeShard(
    const ShardEntries& entries,
    const ShardIndexParameters& shard_index_parameters);

}  // namespace zarr3_sharding_indexed
namespace internal_json_binding {
template <>
constexpr inline auto
    DefaultBinder<zarr3_sharding_indexed::ShardIndexLocation> =
        zarr3_sharding_indexed::ShardIndexLocationJsonBinder;
}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_SHARD_FORMAT_H_
