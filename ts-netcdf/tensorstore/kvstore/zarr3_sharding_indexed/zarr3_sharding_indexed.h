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

#ifndef TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_ZARR_SHARDING_INDEXED_H_
#define TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_ZARR_SHARDING_INDEXED_H_

/// KvStore adapter for the zarr v3 sharding_indexed codec.
///
/// https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html
///
/// This adapter operates on a *single* key within a base kvstore.
///
/// For the purpose of this kvstore adapter, one key in the base kvstore
/// corresponds to a single *shard*.  Each shard contains a fixed-size grid of
/// *entries*, where each entry is simply a byte string.  The size of the grid
/// is given by a `grid_shape` parameter that must be specified when opening the
/// kvstore.
///
/// In the stored representation of the shard, each entry is stored directly as
/// a single contiguous byte range within the shard, but may be at an arbitrary
/// offset.  At the end of the shard is the shard index.
///
/// The shard index is logically represented as a uint64 array, of shape
/// `grid_shape + [2]`.  The inner dimension ranges of `[offset, length]`.
///
/// In the stored representation of the shard, the shard index is encoded using
/// a chain of regular zarr v3 codecs, specified by the `index_codecs`
/// parameter.  It is required that the resultant encoded shard index
/// representation have a fixed size in bytes, as a function of the `grid_shape`
/// and `index_codecs`.  This imposes additional restrictions on the allowed
/// codecs, but allows it to be read with a single suffix-length byte range
/// request.
///
/// To read an entry, the shard index must first be read and decoded, and then
/// the byte range indicated by the shard index is read.  Depending on the cache
/// pool configuration, the shard index may be cached to reduce overhead for
/// repeated read requests to the same shard.
///
/// To write an entry or otherwise make any changes to a shard, the entire shard
/// is re-written.

#include <stdint.h>

#include <string>
#include <string_view>

#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"  // IWYU pragma: export
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace zarr3_sharding_indexed {

struct ShardedKeyValueStoreParameters {
  kvstore::DriverPtr base_kvstore;
  std::string base_kvstore_path;
  Executor executor;
  internal::CachePool::WeakPtr cache_pool;
  ShardIndexParameters index_params;
};

kvstore::DriverPtr GetShardedKeyValueStore(
    ShardedKeyValueStoreParameters&& parameters);

}  // namespace zarr3_sharding_indexed
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZARR_SHARDING_INDEXED_ZARR_SHARDING_INDEXED_H_
