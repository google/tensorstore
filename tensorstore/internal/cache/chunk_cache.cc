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

#include "tensorstore/internal/cache/chunk_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/chunk_receiver_utils.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/grid_partition_iterator.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/regular_grid.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal_metrics::MetricMetadata;

#ifndef TENSORSTORE_INTERNAL_CHUNK_CACHE_DEBUG
#define TENSORSTORE_INTERNAL_CHUNK_CACHE_DEBUG 0
#endif

namespace tensorstore {
namespace internal {

auto& num_writes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/cache/chunk_cache/writes",
    MetricMetadata("Number of writes to ChunkCache."));
auto& num_reads = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/cache/chunk_cache/reads",
    MetricMetadata("Number of reads from ChunkCache."));

namespace {

/// Returns `true` if all components of `node` have been fully overwritten.
///
/// \param node Non-null pointer to transaction node.
bool IsFullyOverwritten(ChunkCache::TransactionNode& node) {
  auto& entry = GetOwningEntry(node);
  const auto& grid = GetOwningCache(entry).grid();
  const auto& component_specs = grid.components;
  const tensorstore::span<const Index> cell_indices = entry.cell_indices();
  for (size_t component_index = 0, num_components = component_specs.size();
       component_index != num_components; ++component_index) {
    if (!node.components()[component_index].write_state.IsFullyOverwritten(
            component_specs[component_index].array_spec,
            grid.GetCellDomain(component_index, cell_indices))) {
      return false;
    }
  }
  return true;
}

/// TensorStore Driver ReadChunk implementation for the chunk cache, for the
/// case of a non-transactional read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly interface.
///
/// The `ChunkCache::Read` operation (when no transaction is specified) proceeds
/// as follows:
///
/// 1. Like the `Write` method, `Read` calls
///    `PartitionIndexTransformOverGrid` to iterate over the set of grid
///    cells (corresponding to cache entries) contained in the range of the
///    user-specified `transform`.
///
/// 2. For each grid cell, `Read` finds the corresponding `ChunkCache::Entry`
///    (creating it if it does not already exist), and constructs a `ReadChunk`
///    object that holds a pointer to the `ChunkCache::Entry` object, the
///    `component_index`, and a transform from a synthetic "chunk" index space
///    to the index space over which the array component is defined (in exactly
///    the same way as the `WriteChunk` is constructed by `Write`).  Storage for
///    the cell (if not previously allocated) is not actually allocated until
///    data is read from the underlying storage.
///
/// 3. Unlike `Write`, however, `Read` does not immediately send the constructed
///    `ReadChunk` to the user-specified `receiver`.  Instead, it calls
///    `AsyncCache::Entry::Read` on the `ChunkCache::Entry` object
///    to wait for data meeting the user-specified `staleness` constraint to be
///    ready.
///
/// 4. If data within the staleness bound is already available for the
///    particular component array of the cell, no actual read operation occurs.
///    Otherwise, this results in a call to the `ChunkCache::Entry::DoRead`
///    method which must be overridden by derived classes, typically using the
///    `KvsBackedCache` mixin.
///
/// 5. The derived class implementation of `DoRead` arranges to call
///    `ChunkCache::Entry::ReadSuccess` when the read has finished successfully.
///    If there is no data for the cell in the underlying storage, the cell is
///    considered to implicitly contain the fill value, but no storage is
///    actually allocated.
///
/// 6. Once the cell data has been updated (if necessary), the `ReadChunk`
///    constructed previously is sent to the user-specified `receiver`.
struct ReadChunkImpl {
  size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;
  bool fill_missing_data_reads;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    // No locks need to be held throughout read operation.  A temporary lock is
    // held in the `BeginRead` method below only while copying the shared_ptr to
    // the immutable cached chunk data.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    auto& grid = GetOwningCache(*entry).grid();
    auto domain = grid.GetCellDomain(component_index, entry->cell_indices());
    SharedArray<const void, dynamic_rank(kMaxRank)> read_array{
        ChunkCache::GetReadComponent(
            AsyncCache::ReadLock<ChunkCache::ReadData>(*entry).data(),
            component_index)};
    if (!fill_missing_data_reads && !read_array.valid()) {
      return absl::NotFoundError(
          tensorstore::StrCat(entry->DescribeChunk(), " is missing"));
    }
    return grid.components[component_index].array_spec.GetReadNDIterable(
        std::move(read_array), domain, std::move(chunk_transform), arena);
  }
};

/// TensorStore Driver ReadChunk implementation for the chunk cache, for the
/// case of a transactional read.
///
/// This implements the `tensorstore::internal::ReadChunk::Impl` Poly interface.
///
/// Unlike a non-transactional read, a transactional read also sees uncommitted
/// writes made previously in the same transaction.
///
/// The process of reading with a transaction proceeds similarly to the process
/// of reading without a transaction, as described above, except that after
/// obtaining the `ChunkCache::Entry` for a cell, `Read` also obtains a
/// transaction node using `GetTransactionNode`, and that node is stored in the
/// `ReadChunk`.
///
/// Additionally, `Read` calls the `ChunkCache::TransactionNode::DoRead` method,
/// rather than `ChunkCache::Entry::DoRead`.
struct ReadChunkTransactionImpl {
  size_t component_index;
  OpenTransactionNodePtr<ChunkCache::TransactionNode> node;
  bool fill_missing_data_reads;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    constexpr auto lock_chunk = [](void* data, bool lock)
                                    ABSL_NO_THREAD_SAFETY_ANALYSIS -> bool {
      auto& node = *static_cast<ChunkCache::TransactionNode*>(data);
      if (lock) {
        // We are allowed to read from a revoked transaction node (and in fact
        // it is necessary in order to avoid potential livelock).  Therefore,
        // this always succeeds unconditionally.
        node.WriterLock();
      } else {
        node.WriterUnlock();
      }
      return true;
    };
    lock_collection.Register(node.get(), +lock_chunk, /*shared=*/true);
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    auto& entry = GetOwningEntry(*node);
    auto& grid = GetOwningCache(entry).grid();
    const auto& component_spec = grid.components[component_index];
    auto& component = node->components()[component_index];
    auto domain = grid.GetCellDomain(component_index, entry.cell_indices());
    SharedArray<const void, dynamic_rank(kMaxRank)> read_array;
    StorageGeneration read_generation;
    // Copy the shared_ptr to the immutable cached chunk data for the node.  If
    // any elements of the chunk have not been overwritten, the cached data is
    // needed to fill them in if they are not already up to date.
    {
      // Note that this acquires a lock on the entry, not the node, and
      // therefore does not conflict with the lock registered with the
      // `LockCollection`.
      AsyncCache::ReadLock<ChunkCache::ReadData> read_lock(*node);
      read_array =
          ChunkCache::GetReadComponent(read_lock.data(), component_index);
      read_generation = read_lock.stamp().generation;
      if (!node->IsUnconditional() &&
          (node->transaction()->mode() & repeatable_read)) {
        TENSORSTORE_RETURN_IF_ERROR(
            node->RequireRepeatableRead(read_generation));
      }
    }
    if (!fill_missing_data_reads && !read_array.valid()) {
      return absl::NotFoundError(
          tensorstore::StrCat(entry.DescribeChunk(), " is missing"));
    }
    return component.GetReadNDIterable(component_spec.array_spec, domain,
                                       std::move(read_array), read_generation,
                                       std::move(chunk_transform), arena);
  }
};

/// TensorStore Driver WriteChunk implementation for the chunk cache.
///
/// This implements the `tensorstore::internal::WriteChunk::Impl` Poly
/// interface.
///
/// The `ChunkCache::Write` operation proceeds as follows:
///
/// There are two separate phases to `Write` operations: copying the data into
/// the cache, and writing the data back to persistent storage.
///
/// Phase I: Copying the data into the cache
/// ----------------------------------------
///
/// 1. `Write` calls `PartitionIndexTransformOverGrid` to iterate over
///    the set of grid cells (corresponding to cache entries) contained in the
///    range of the user-specified `transform`.
///
/// 2. For each grid cell, `Write` finds the corresponding `ChunkCache::Entry`
///    (creating it if it does not already exist), obtains a new or existing
///    transaction node for the entry, and immediately (synchronously) sends to
///    the user-specified `receiver` a `WriteChunk` object that holds a pointer
///    to the `ChunkCache::TransactionNode` object, the `component_index`, and a
///    transform from a synthetic "chunk" index space to the index space over
///    which the array component is defined.  (This transform is obtained by
///    simply composing the user-specified `transform` with the `cell_transform`
///    computed by `PartitionIndexTransformOverGrid`.  Along with the
///    `WriteChunk` object, the `cell_transform` is also sent to the `receiver`;
///    the `receiver` uses this `cell_transform` to convert from the domain of
///    the original input domain (i.e. the input domain of the user-specified
///    `transform`) to the synthetic "chunk" index space that must be used to
///    write to the chunk.  Storage for the cell (if not previously allocated)
///    is not actually allocated until the user writes to the chunk.
///
/// 3. Writes made to `WriteChunk` objects sent to the `receiver` result in
///    calls to `AsyncWriteArray`, which provides the necessary tracking to
///    support optimistic (atomic read-modify-write) concurrency.  Writeback is
///    deferred until the explicit transaction is committed (in the case of a
///    transactional write) or the individual implicit transactions associated
///    with each entry are committed (in the case of a non-transactional write).
///
///    In the non-transactional case, the `commit_future` returned by `Write` is
///    linked to all of the futures associated with the implicit transactions,
///    and calling `Force` on the `commit_future` forces all of the implicit
///    transactions to be committed.  The implicit transactions may also be
///    committed automatically due to memory pressure on the `CachePool`.
///
/// Phase II: Writeback to persistent storage
/// -----------------------------------------
///
/// 4. The derived class is responsible for overriding
///    `ChunkCache::TransactionNode::Commit`, typically via the `KvsBackedCache`
///    mixin.  The `Commit` implementation calls
///    `ChunkCache::TransactionNode::DoApply`, which provides updated component
///    arrays with all writes applied.
///
/// 5. `ChunkCache::TransactionNode::DoApply` calls
///    `ChunkCache::TransactionNode::Read` if the cell has not been completely
///    locally overwritten.  If there is not already a cached read result, this
///    results in a call to `ChunkCache::TransactionNode::DoRead`, which is
///    responsible for fetching and decoding the existing data for the cell, if
///    available, and is typically implemented via the `KvsBachedCache` mixin.
///    `DoApply` relies on `AsyncWriteArray` to merge the updated component
///    arrays (or the fill value arrays if there is no existing data) with the
///    local modifications.
struct WriteChunkImpl {
  size_t component_index;
  OpenTransactionNodePtr<ChunkCache::TransactionNode> node;
  bool store_data_equal_to_fill_value;

  absl::Status operator()(internal::LockCollection& lock_collection) {
    constexpr auto lock_chunk = [](void* data, bool lock)
                                    ABSL_NO_THREAD_SAFETY_ANALYSIS -> bool {
      auto& node = *static_cast<ChunkCache::TransactionNode*>(data);
      if (lock) {
        return node.try_lock();
      } else {
        node.WriterUnlock();
        return true;
      }
    };
    if (node->IsRevoked()) {
      OpenTransactionPtr transaction(node->transaction());
      TENSORSTORE_ASSIGN_OR_RETURN(
          node, GetTransactionNode(GetOwningEntry(*node), transaction));
    }
    lock_collection.Register(node.get(), +lock_chunk, /*shared=*/false);
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    auto& entry = GetOwningEntry(*node);
    auto& grid = GetOwningCache(entry).grid();
    const auto& component_spec = grid.components[component_index];
    auto domain = grid.GetCellDomain(component_index, entry.cell_indices());
    node->MarkSizeUpdated();
    auto& async_write_array = node->components()[component_index];
    if (store_data_equal_to_fill_value) {
      async_write_array.write_state.store_if_equal_to_fill_value = true;
    }
    return async_write_array.BeginWrite(component_spec.array_spec, domain,
                                        std::move(chunk_transform), arena);
  }

  WriteChunk::EndWriteResult operator()(WriteChunk::EndWrite,
                                        IndexTransformView<> chunk_transform,
                                        bool success, Arena* arena) const {
    auto& entry = GetOwningEntry(*node);
    auto& grid = GetOwningCache(entry).grid();
    const auto& component_spec = grid.components[component_index];
    auto domain = grid.GetCellDomain(component_index, entry.cell_indices());
    node->components()[component_index].EndWrite(
        component_spec.array_spec, domain, chunk_transform, success, arena);
    node->is_modified = true;
    if (IsFullyOverwritten(*node)) {
      node->SetUnconditional();
    }
    return {node->OnModified(), node->transaction()->future()};
  }

  bool operator()(WriteChunk::WriteArray, IndexTransformView<> chunk_transform,
                  WriteChunk::GetWriteSourceArrayFunction get_source_array,
                  Arena* arena,
                  WriteChunk::EndWriteResult& end_write_result) const {
    auto& entry = GetOwningEntry(*node);
    auto& grid = GetOwningCache(entry).grid();
    const auto& component_spec = grid.components[component_index];
    auto domain = grid.GetCellDomain(component_index, entry.cell_indices());
    using WriteArraySourceCapabilities =
        AsyncWriteArray::WriteArraySourceCapabilities;
    auto& async_write_array = node->components()[component_index];
    if (store_data_equal_to_fill_value) {
      async_write_array.write_state.store_if_equal_to_fill_value = true;
    }
    auto status = async_write_array.WriteArray(
        component_spec.array_spec, domain, chunk_transform,
        [&]() -> Result<std::pair<TransformedSharedArray<const void>,
                                  WriteArraySourceCapabilities>> {
          // Translate source array and source restriction into write array and
          // `WriteArraySourceCapabilities`.
          TENSORSTORE_ASSIGN_OR_RETURN(auto info, get_source_array());
          auto source_restriction = std::get<1>(info);
          WriteArraySourceCapabilities source_capabilities;
          switch (source_restriction) {
            case cannot_reference_source_data:
              source_capabilities = WriteArraySourceCapabilities::kCannotRetain;
              break;
            case can_reference_source_data_indefinitely:
              source_capabilities = WriteArraySourceCapabilities::
                  kImmutableAndCanRetainIndefinitely;
              break;
          }
          return {std::in_place, std::move(std::get<0>(info)),
                  source_capabilities};
        });
    if (!status.ok()) {
      if (absl::IsCancelled(status)) return false;
      end_write_result = {status};
      return true;
    }
    node->is_modified = true;
    node->SetUnconditional();
    end_write_result = {node->OnModified(), node->transaction()->future()};
    return true;
  }
};

}  // namespace

void ChunkCache::Read(ReadRequest request, ReadChunkReceiver receiver) {
  assert(request.component_index >= 0 &&
         request.component_index < grid().components.size());
  const auto& component_spec = grid().components[request.component_index];
  // Shared state used while `Read` is in progress.
  using ReadOperationState = ChunkOperationState<ReadChunk>;

  assert(component_spec.chunked_to_cell_dimensions.size() ==
         grid().chunk_shape.size());
  auto state = MakeIntrusivePtr<ReadOperationState>(std::move(receiver));
  internal_grid_partition::RegularGridRef regular_grid{grid().chunk_shape};
  auto status = PartitionIndexTransformOverGrid(
      component_spec.chunked_to_cell_dimensions, regular_grid,
      request.transform,
      [&](tensorstore::span<const Index> grid_cell_indices,
          IndexTransformView<> cell_transform) {
        if (state->cancelled()) {
          return absl::CancelledError("");
        }
        num_reads.Increment();
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto cell_to_source,
            ComposeTransforms(request.transform, cell_transform));
        auto entry = GetEntryForGridCell(*this, grid_cell_indices);
        // Arrange to call `set_value` on the receiver with a `ReadChunk`
        // corresponding to this grid cell once the read request completes
        // successfully.
        ReadChunk chunk;
        chunk.transform = std::move(cell_to_source);
        Future<const void> read_future;
        const auto get_cache_read_request = [&] {
          AsyncCache::AsyncCacheReadRequest cache_request;
          cache_request.staleness_bound = request.staleness_bound;
          cache_request.batch = request.batch;
          return cache_request;
        };
        if (request.transaction) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto node, GetTransactionNode(*entry, request.transaction));
          read_future = node->IsUnconditional()
                            ? MakeReadyFuture()
                            : node->Read(get_cache_read_request());
          chunk.impl =
              ReadChunkTransactionImpl{request.component_index, std::move(node),
                                       request.fill_missing_data_reads};
        } else {
          read_future = entry->Read(get_cache_read_request());
          chunk.impl = ReadChunkImpl{request.component_index, std::move(entry),
                                     request.fill_missing_data_reads};
        }
        LinkValue(
            [state, chunk = std::move(chunk),
             cell_transform = IndexTransform<>(cell_transform)](
                Promise<void> promise, ReadyFuture<const void> future) mutable {
              execution::set_value(state->shared_receiver->receiver,
                                   std::move(chunk), std::move(cell_transform));
            },
            state->promise, std::move(read_future));
        return absl::OkStatus();
      });
  if (!status.ok()) {
    state->SetError(std::move(status));
  }
}

void ChunkCache::Write(WriteRequest request, WriteChunkReceiver receiver) {
  assert(request.component_index >= 0 &&
         request.component_index < grid().components.size());
  // In this implementation, chunks are always available for writing
  // immediately.  The entire stream of chunks is sent to the receiver before
  // this function returns.
  const auto& component_spec = grid().components[request.component_index];
  std::atomic<bool> cancelled{false};
  execution::set_starting(receiver, [&cancelled] { cancelled = true; });
  internal_grid_partition::RegularGridRef regular_grid{grid().chunk_shape};
  absl::Status status = PartitionIndexTransformOverGrid(
      component_spec.chunked_to_cell_dimensions, regular_grid,
      request.transform,
      [&](tensorstore::span<const Index> grid_cell_indices,
          IndexTransformView<> cell_transform) {
        if (cancelled) return absl::CancelledError("");
        num_writes.Increment();
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto cell_to_dest,
            ComposeTransforms(request.transform, cell_transform));
        ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_CHUNK_CACHE_DEBUG)
            << "grid_cell_indices=" << grid_cell_indices
            << ", request.transform=" << request.transform
            << ", cell_transform=" << cell_transform
            << ", cell_to_dest=" << cell_to_dest;
        auto entry = GetEntryForGridCell(*this, grid_cell_indices);
        auto transaction_copy = request.transaction;
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto node, GetTransactionNode(*entry, transaction_copy));
        execution::set_value(
            receiver,
            WriteChunk{WriteChunkImpl{request.component_index, std::move(node),
                                      request.store_data_equal_to_fill_value},
                       std::move(cell_to_dest)},
            IndexTransform<>(cell_transform));
        return absl::OkStatus();
      });
  if (!status.ok()) {
    execution::set_error(receiver, status);
  } else {
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

Future<const void> ChunkCache::DeleteCell(
    tensorstore::span<const Index> grid_cell_indices,
    internal::OpenTransactionPtr transaction) {
  return GetEntryForGridCell(*this, grid_cell_indices)->Delete(transaction);
}

absl::Status ChunkCache::TransactionNode::Delete() {
  UniqueWriterLock lock(*this);
  this->MarkSizeUpdated();
  this->is_modified = true;
  auto& entry = GetOwningEntry(*this);
  const tensorstore::span<const Index> cell_indices = entry.cell_indices();
  const auto& grid = GetOwningCache(entry).grid();
  for (Index component_index = 0, num_components = grid.components.size();
       component_index != num_components; ++component_index) {
    const auto& component_spec = grid.components[component_index];
    auto domain = grid.GetCellDomain(component_index, cell_indices);
    // There is no need to check the reference count of the component data array
    // (i.e. to check for a concurrent read) because this doesn't modify the
    // data array, it just resets the data pointer to `nullptr`.
    components()[component_index].write_state.WriteFillValue(
        component_spec.array_spec, domain);
  }
  SetUnconditional();
  return OnModified();
}

Future<const void> ChunkCache::Entry::Delete(OpenTransactionPtr transaction) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                               GetTransactionNode(*this, transaction));
  TENSORSTORE_RETURN_IF_ERROR(node->Delete());
  return node->transaction()->future();
}

size_t ChunkCache::TransactionNode::ComputeWriteStateSizeInBytes() {
  size_t total = 0;
  const auto component_specs = this->component_specs();
  for (size_t component_index = 0;
       component_index < static_cast<size_t>(component_specs.size());
       ++component_index) {
    auto& component_spec = component_specs[component_index];
    total +=
        this->components()[component_index].write_state.EstimateSizeInBytes(
            component_spec.array_spec, component_spec.chunk_shape);
  }
  return total;
}

size_t ChunkCache::Entry::ComputeReadDataSizeInBytes(const void* read_data) {
  const ReadData* components = static_cast<const ReadData*>(read_data);
  size_t total = 0;
  auto component_specs = this->component_specs();
  for (size_t component_index = 0;
       component_index < static_cast<size_t>(component_specs.size());
       ++component_index) {
    auto& component_spec = component_specs[component_index];
    total += component_spec.array_spec.EstimateReadStateSizeInBytes(
        components[component_index].valid(), component_spec.chunk_shape);
  }
  return total;
}

ChunkCache::WritebackSnapshot::WritebackSnapshot(
    TransactionNode& node, AsyncCache::ReadView<ReadData> read_state) {
  auto& entry = GetOwningEntry(node);
  auto& grid = GetOwningCache(entry).grid();
  const tensorstore::span<const Index> cell_indices = entry.cell_indices();
  for (size_t component_i = 0; component_i < grid.components.size();
       ++component_i) {
    const auto& component_spec = grid.components[component_i];
    auto& component = node.components()[component_i];
    auto domain = grid.GetCellDomain(component_i, cell_indices);
    auto component_snapshot = component.GetArrayForWriteback(
        component_spec.array_spec, domain,
        GetReadComponent(read_state.data(), component_i),
        read_state.stamp().generation);
    if (component_snapshot.must_store) {
      if (!new_read_data_) {
        new_read_data_ = internal::make_shared_for_overwrite<ReadData[]>(
            grid.components.size());
      }
      new_read_data_.get()[component_i] = std::move(component_snapshot.array);
    }
  }
}

ChunkCache::TransactionNode::TransactionNode(Entry& entry)
    : AsyncCache::TransactionNode(entry) {
  const auto& component_specs = GetOwningCache(entry).grid().components;
  components_.reserve(component_specs.size());
  for (size_t i = 0; i < component_specs.size(); ++i) {
    components_.emplace_back(component_specs[i].rank());
  }
}

absl::Status ChunkCache::TransactionNode::OnModified() {
  return absl::OkStatus();
}

void ChunkCache::TransactionNode::DoApply(ApplyOptions options,
                                          ApplyReceiver receiver) {
  if (options.apply_mode == ApplyOptions::kValidateOnly) {
    execution::set_value(
        receiver, ReadState{{}, TimestampedStorageGeneration::Unconditional()});
    return;
  }
  auto continuation = WithExecutor(
      GetOwningCache(*this).executor(),
      [this, receiver = std::move(receiver),
       specify_unchanged =
           options.apply_mode == ApplyOptions::kSpecifyUnchanged](
          tensorstore::ReadyFuture<const void> future) mutable {
        if (!future.result().ok()) {
          return execution::set_error(receiver, future.result().status());
        }
        AsyncCache::ReadState read_state;
        if (this->IsUnconditional() ||
            (!this->is_modified && !specify_unchanged)) {
          read_state.stamp = TimestampedStorageGeneration::Unconditional();
        } else {
          read_state = AsyncCache::ReadLock<void>(*this).read_state();
        }
        if (is_modified) {
          // Protect against concurrent calls to `DoApply`, since this may
          // modify the write arrays to incorporate the read state.
          UniqueWriterLock<AsyncCache::TransactionNode> lock(*this);
          WritebackSnapshot snapshot(
              *this, AsyncCache::ReadView<ReadData>(read_state));
          read_state.data = std::move(snapshot.new_read_data());
          read_state.stamp.generation.MarkDirty();
        }
        execution::set_value(receiver, std::move(read_state));
      });
  if (this->IsUnconditional() ||
      (!this->is_modified &&
       options.apply_mode != ApplyOptions::kSpecifyUnchanged)) {
    continuation(MakeReadyFuture());
  } else {
    this->Read({options.staleness_bound})
        .ExecuteWhenReady(std::move(continuation));
  }
}

void ChunkCache::TransactionNode::InvalidateReadState() {
  AsyncCache::TransactionNode::InvalidateReadState();
  for (auto& component : components()) {
    component.InvalidateReadState();
  }
}

std::string ChunkCache::Entry::DescribeChunk() {
  return tensorstore::StrCat("chunk ", this->cell_indices());
}

}  // namespace internal
}  // namespace tensorstore
