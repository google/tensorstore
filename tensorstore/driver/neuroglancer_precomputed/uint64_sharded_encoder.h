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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_ENCODER_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_ENCODER_H_

/// \file
/// Support for encoding the neuroglancer_uint64_sharded_v1 format.
///
/// This format maps uint64 keys to byte strings, and supports lookups of
/// arbitrary keys with only 3 reads.
///
/// See description of format here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#sharded-format

#include <cstdint>
#include <utility>
#include <vector>

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

/// Encodes a minishard index.
///
/// The format is described here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#minishard-index-format
absl::Cord EncodeMinishardIndex(
    span<const MinishardIndexEntry> minishard_index);

/// Encodes a shard index.
///
/// The format is described here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#shardindex-index-file-format
absl::Cord EncodeShardIndex(span<const ShardIndexEntry> shard_index);

/// Class that may be used to sequentially encode a single shard.
///
/// The keys must already be sorted by minishard number.
///
/// Example usage:
///
///     absl::Cord out;
///     ShardEncoder shard_encoder(sharding_spec, out);
///     for (const auto &entry : entries) {
///       // Optionally write additional unindexed data
///       TENSORSTORE_ASSIGN_OR_RETURN(
///           auto offsets,
///           shard_writer.WriteUnindexedEntry(entry.minishard,
///                                            extra_data,
///                                            /*compress=*/false));
///       auto entry_data = MakeEntryData(entry, offsets);
///       // Write the entry.
///       TENSORSTORE_RETURN_IF_ERROR(shard_writer.WriteIndexedEntry(
///           entry.minishard,
///           entry.chunk_id,
///           entry_data,
///           /*compress=*/true));
///     }
///     TENSORSTORE_ASSIGN_OR_RETURN(auto shard_index, shard_writer.Finalize());
class ShardEncoder {
 public:
  using WriteFunction = std::function<absl::Status(const absl::Cord& buffer)>;

  /// Constructs a shard encoder.
  ///
  /// \param sharding_spec The sharding specification.
  /// \param write_function Function called to write encoded data for the shard.
  explicit ShardEncoder(const ShardingSpec& sharding_spec,
                        WriteFunction write_function);

  /// Same as above, but appends output to `out`.
  explicit ShardEncoder(const ShardingSpec& sharding_spec, absl::Cord& out);

  /// Writes a single chunk.
  ///
  /// The chunk will be included in the minishard index.
  ///
  /// \param minishard The minishard number, must be >= any previous `minishard`
  ///     values supplied to `WriteIndexedEntry` or `WriteUnindexedEntry`.
  /// \param chunk_id The chunk id, must map to `minishard` and must be distinct
  ///     from any previous `chunk_id` supplied to `WriteIndexedEntry`.
  /// \param data The chunk data to write.
  /// \param compress Specifies whether to honor the `data_compression`
  ///     specified in `sharding_spec`.  This should normally be `true` unless
  ///     the format is being used in a special way, e.g. to write the
  ///     multiscale mesh fragment data, or for copying already compressed data
  ///     from an existing shard.
  /// \pre `Finalize()` was not called previously, and no prior method call
  ///     returned an error.
  Status WriteIndexedEntry(std::uint64_t minishard, ChunkId chunk_id,
                           const absl::Cord& data, bool compress);

  /// Writes an additional chunk of data to the shard data file, but does not
  /// include it in the index under a particular `chunk_id` key.
  ///
  /// This is used in the implementation of `WriteIndexedEntry`, and can also be
  /// called directly in order to store additional data that may be referenced
  /// by another chunk via the returned offsets.  This functionality is used by
  /// the mesh format, for example, to store the mesh fragment data.
  ///
  /// \param minishard The minishard number, must be >= any previous `minishard`
  ///     values supplied to `WriteIndexedEntry` or `WriteUnindexedEntry`.
  /// \param data The chunk data to write.
  /// \param compress Specifies whether to honor the `data_compression`
  ///     specified in `sharding_spec`.
  /// \return The location of the chunk on success.
  /// \pre `Finalize()` was not called previously, and no prior method call
  ///     returned an error.
  Result<ByteRange> WriteUnindexedEntry(std::uint64_t minishard,
                                        const absl::Cord& data, bool compress);

  /// Finalizes the shard data file and returns the encoded shard index file.
  ///
  /// \pre `Finalize()` was not called previously, and no prior method call
  ///     returned an error.
  Result<absl::Cord> Finalize();

  /// Returns the sharding specification.
  const ShardingSpec& sharding_spec() const { return sharding_spec_; }

  ~ShardEncoder();

 private:
  /// Finalizes the current minishard, writing the minishard index for it if it
  /// is non-empty.
  ///
  /// \pre `Finalize()` was not called previously.
  Status FinalizeMinishard();

  ShardingSpec sharding_spec_;
  WriteFunction write_function_;

  /// The minishard index for the current minishard.
  std::vector<MinishardIndexEntry> minishard_index_;

  /// The shard index specifying the offsets of all previously-written minishard
  /// indices.
  std::vector<ShardIndexEntry> shard_index_;

  /// The current minishard number, initially 0.
  std::uint64_t cur_minishard_;

  /// The number of bytes that have been written to the shard data file.
  std::uint64_t data_file_offset_;
};

/// Encodes a full shard from a list of chunks.
///
/// \param chunks The chunks to include, must be ordered by minishard index and
///     then by chunk id.
std::optional<absl::Cord> EncodeShard(const ShardingSpec& spec,
                                      span<const EncodedChunk> chunks);

absl::Cord EncodeData(const absl::Cord& input,
                      ShardingSpec::DataEncoding encoding);

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_ENCODER_H_
