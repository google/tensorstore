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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_H_

/// \file
/// Common utilities for reading or writing the neuroglancer_uint64_sharded_v1
/// format.
///
/// Refer to the specification here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#sharded-format

#include <ostream>
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

/// Specifies sharded storage parameters.
///
/// Refer to the specification here:
/// https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#sharding-specification
class ShardingSpec {
 public:
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ShardingSpec,
                                          internal::json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  /// Specifies the hash function used to compute the minishard and shard
  /// numbers.
  enum class HashFunction {
    identity,
    murmurhash3_x86_128,
  };

  friend std::ostream& operator<<(std::ostream& os, HashFunction x);

  friend void to_json(::nlohmann::json& out,  // NOLINT
                      HashFunction x);

  enum class DataEncoding {
    raw,
    gzip,
  };

  friend std::ostream& operator<<(std::ostream& os, DataEncoding x);

  friend std::ostream& operator<<(std::ostream& os, const ShardingSpec& x);

  ShardingSpec() = default;

  ShardingSpec(HashFunction hash_function, int preshift_bits,
               int minishard_bits, int shard_bits, DataEncoding data_encoding,
               DataEncoding minishard_index_encoding)
      : hash_function(hash_function),
        preshift_bits(preshift_bits),
        minishard_bits(minishard_bits),
        shard_bits(shard_bits),
        data_encoding(data_encoding),
        minishard_index_encoding(minishard_index_encoding) {}

  HashFunction hash_function;
  int preshift_bits;
  int minishard_bits;
  int shard_bits;
  DataEncoding data_encoding = DataEncoding::raw;
  DataEncoding minishard_index_encoding = DataEncoding::raw;

  std::uint64_t num_shards() const {
    return static_cast<std::uint64_t>(1) << shard_bits;
  }

  std::uint64_t num_minishards() const {
    return static_cast<std::uint64_t>(1) << minishard_bits;
  }

  friend bool operator==(const ShardingSpec& a, const ShardingSpec& b);
  friend bool operator!=(const ShardingSpec& a, const ShardingSpec& b) {
    return !(a == b);
  }
};

/// Returns the data path for the specified shard.
std::string GetShardKey(const ShardingSpec& sharding_spec,
                        std::string_view prefix, std::uint64_t shard_number);

struct ChunkId {
  std::uint64_t value;
};

/// Hashes a pre-shifted 64-bit key with the specified hash function.
std::uint64_t HashChunkId(ShardingSpec::HashFunction h, std::uint64_t key);

struct ChunkCombinedShardInfo {
  std::uint64_t shard_and_minishard;
};

struct ChunkSplitShardInfo {
  std::uint64_t minishard;
  std::uint64_t shard;
};

ChunkCombinedShardInfo GetChunkShardInfo(const ShardingSpec& sharding_spec,
                                         ChunkId chunk_id);

ChunkSplitShardInfo GetSplitShardInfo(const ShardingSpec& sharding_spec,
                                      ChunkCombinedShardInfo combined_info);

/// In-memory representation of a minishard index entry.
///
/// Specifies the start and end offsets of a particular `chunk_id` within the
/// shard data file.
struct MinishardIndexEntry {
  /// Chunk identifier.
  ChunkId chunk_id;

  /// Location of chunk data.
  ByteRange byte_range;

  friend bool operator==(const MinishardIndexEntry& a,
                         const MinishardIndexEntry& b) {
    return a.chunk_id.value == b.chunk_id.value && a.byte_range == b.byte_range;
  }
  friend bool operator!=(const MinishardIndexEntry& a,
                         const MinishardIndexEntry& b) {
    return !(a == b);
  }
};

/// Specifies the start and end offsets of the minishard index for a particular
/// minishard.
using ShardIndexEntry = ByteRange;

/// Returns the size in bytes of the shard index.
std::uint64_t ShardIndexSize(const ShardingSpec& sharding_spec);

/// Converts a byte range relative to the end of the shard index to be relative
/// to the start of the shard file.
Result<ByteRange> GetAbsoluteShardByteRange(ByteRange relative_range,
                                            const ShardingSpec& sharding_spec);

struct MinishardAndChunkId {
  uint64_t minishard;
  ChunkId chunk_id;
  friend bool operator<(const MinishardAndChunkId& a,
                        const MinishardAndChunkId& b) {
    return (a.minishard < b.minishard) ||
           (a.minishard == b.minishard && a.chunk_id.value < b.chunk_id.value);
  }
  friend bool operator==(const MinishardAndChunkId& a,
                         const MinishardAndChunkId& b) {
    return a.minishard == b.minishard && a.chunk_id.value == b.chunk_id.value;
  }

  friend bool operator!=(const MinishardAndChunkId& a,
                         const MinishardAndChunkId& b) {
    return !(a == b);
  }
};

struct EncodedChunk {
  MinishardAndChunkId minishard_and_chunk_id;
  /// Chunk data, compressed according to the `DataEncoding` value.
  absl::Cord encoded_data;
};

using EncodedChunks = std::vector<EncodedChunk>;

/// Finds a chunk in an ordered list of chunks.
const EncodedChunk* FindChunk(span<const EncodedChunk> chunks,
                              MinishardAndChunkId minishard_and_chunk_id);

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_H_
