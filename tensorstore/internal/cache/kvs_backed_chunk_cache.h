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

#ifndef TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CHUNK_CACHE_H_
#define TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CHUNK_CACHE_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Integrates `ChunkCache` with `KvsBackedCache`.
///
/// Derived classes must implement `DecodeChunk` and `EncodeChunk`.
class KvsBackedChunkCache
    : public internal::KvsBackedCache<KvsBackedChunkCache,
                                      internal::ChunkCache> {
 public:
  using Base =
      internal::KvsBackedCache<KvsBackedChunkCache, internal::ChunkCache>;

  using Base::Base;

  virtual std::string GetChunkStorageKey(span<const Index> cell_indices) = 0;

  /// Decodes a data chunk.
  ///
  /// \param data The encoded chunk data.
  /// \returns On success, returns a decoded array for each component.  The
  ///     shape of each decoded array `i` must equal
  ///     `grid.components[i].cell_shape()`, where
  ///     `grid = GetChunkGridSpecification(metadata)`.
  virtual Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) = 0;

  /// Encodes a data chunk.
  ///
  /// \param component_arrays Chunk data for each component.
  /// \pre `component_arrays[i].shape() == grid.components[i].cell_shape()`,
  ///     where `grid = GetChunkGridSpecification(metadata)`.
  virtual Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) = 0;

  // The members below are implementation details not relevant to derived class
  // driver implementations.

  class Entry : public Base::Entry {
   public:
    using OwningCache = KvsBackedChunkCache;
    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override;
    void DoEncode(std::shared_ptr<const ReadData> data,
                  EncodeReceiver receiver) override;
    std::string GetKeyValueStoreKey() override;
  };

  Entry* DoAllocateEntry() override { return new Entry; }
  size_t DoGetSizeofEntry() override { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(
      AsyncCache::Entry& entry) override {
    return new TransactionNode(static_cast<Entry&>(entry));
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CHUNK_CACHE_H_
