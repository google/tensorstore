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

#ifndef TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_KEY_VALUE_STORE_H_
#define TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_KEY_VALUE_STORE_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

using GetMaxChunksPerShardFunction =
    std::function<std::uint64_t(std::uint64_t)>;

/// Provides read/write access to the Neuroglancer precomputed sharded format on
/// top of a base `KeyValueStore` that supports byte range reads.
///
/// Refer to the specification here:
/// https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md
///
/// The returned `KeyValueStore` requires keys to be 8 byte strings specifying
/// the uint64 chunk id in native endian.  Note that the native endian encoding
/// of the chunk ids is used only for the `KeyValueStore` interface; it has no
/// bearing on the format of the stored data.
///
/// Currently, only the volume and skeleton formats are supported.  The mesh
/// format is not supported, since the mesh fragment data is not directly
/// accessed using a uint64 key.
///
/// Both reading and writing are supported.
///
/// Read requests require a maximum of 3 reads to the underlying `base_kvstore`:
///
/// 1. Retrieve the shard index entry.  Specifies the byte range of the
///    minishard index, if present.
///
/// 2. Retrieve the minishard index (if present) Specifies the byte range of the
///    chunk data, if present.
///
/// 3. Retrieve the chunk data (if present).
///
/// However, the minshard indexes are cached in the specified `cache_pool`, and
/// therefore subsequent reads within the same minishard require only a single
/// read to the underlying `base_kvstore`.
///
/// Writing is supported, and concurrent writes from multiple machines are
/// safely handled provided that the underlying `KeyValueStore` supports
/// conditional operations.  However, unless used in a restricted manner, writes
/// can be very inefficient:
///
/// 1. Since a shard can only be updated by rewriting it entirely, it is most
///    efficient to group write operations by shard, issue all writes to a given
///    shard without forcing the returned futures, and only then forcing the
///    returned futures to commit.
///
/// 2. The temporary memory required to write a shard is 2 to 3 times the size
///    of the shard.  It is therefore advised that the shards be kept as small
///    as possible (while still avoiding an excess number of objects in the
///    underlying `KeyValueStore`) if writes are to be performed.
///
/// \param base_kvstore The underlying `KeyValueStore` that holds the shard
///     files.
/// \param executor Executor to use for data copying and encoding (not waiting
///     on I/O).
/// \param key_prefix Prefix of the sharded database within `base_kvstore`.
/// \param sharding_spec Sharding specification.
/// \param cache_pool The cache pool for the minishard index cache and for the
///     shard write cache.
/// \param get_max_chunks_per_shard Optional.  Specifies function that computes
///     the maximum number of chunks that may be assigned to the shard.  When
///     writing a shard where the number of new chunks is equal to the maximum
///     for the shard, an unconditional write will be used, which may avoid
///     significant additional data transfer.  If not specified, the maximum is
///     assumed to be unknown and all writes will be conditional.  This is used
///     by the `neuroglancer_precomputed` volume driver to allow shard-aligned
///     writes to be performed unconditionally, in the case where a shard
///     corresponds to a rectangular region.
KeyValueStore::Ptr GetShardedKeyValueStore(
    KeyValueStore::Ptr base_kvstore, Executor executor, std::string key_prefix,
    const ShardingSpec& sharding_spec, internal::CachePool::WeakPtr cache_pool,
    GetMaxChunksPerShardFunction get_max_chunks_per_shard = {});

/// Returns a key suitable for use with a `KeyValueStore` returned from
/// `GetShardedKeyValueStore`.
///
/// The chunk id is encoded as an 8-byte `uint64be` value.
std::string ChunkIdToKey(ChunkId chunk_id);

/// Inverse of `ChunkIdToKey`.
std::optional<ChunkId> KeyToChunkId(std::string_view key);

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NEUROGLANCER_PRECOMPUTED_UINT64_SHARDED_KEY_VALUE_STORE_H_
