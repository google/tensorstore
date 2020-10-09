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

#include "tensorstore/internal/chunk_cache.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/async_cache.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/rank.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

ChunkGridSpecification::Component::Component(SharedArray<const void> fill_value,
                                             Box<> component_bounds)
    : internal::AsyncWriteArray::Spec(std::move(fill_value),
                                      std::move(component_bounds)) {
  chunked_to_cell_dimensions.resize(rank());
  std::iota(chunked_to_cell_dimensions.begin(),
            chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));
}

ChunkGridSpecification::Component::Component(
    SharedArray<const void> fill_value, Box<> component_bounds,
    std::vector<DimensionIndex> chunked_to_cell_dimensions)
    : internal::AsyncWriteArray::Spec(std::move(fill_value),
                                      std::move(component_bounds)),
      chunked_to_cell_dimensions(std::move(chunked_to_cell_dimensions)) {}

ChunkGridSpecification::ChunkGridSpecification(Components components_arg)
    : components(std::move(components_arg)) {
  assert(components.size() > 0);
  // Extract the chunk shape from the cell shape of the first component.
  chunk_shape.resize(components[0].chunked_to_cell_dimensions.size());
  for (DimensionIndex i = 0;
       i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
    chunk_shape[i] =
        components[0].shape()[components[0].chunked_to_cell_dimensions[i]];
  }
  // Verify that the extents of the chunked dimensions are the same for all
  // components.
#if !defined(NDEBUG)
  for (const auto& component : components) {
    assert(component.chunked_to_cell_dimensions.size() == chunk_shape.size());
    for (DimensionIndex i = 0;
         i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
      assert(chunk_shape[i] ==
             component.shape()[component.chunked_to_cell_dimensions[i]]);
    }
  }
#endif  // !defined(NDEBUG)
}

namespace {

/// Computes the origin of a cell for a particular component array at the
/// specified grid position.
///
/// \param spec Grid specification.
/// \param component_spec Component specification.
/// \param cell_indices Pointer to array of length `spec.rank()` specifying the
///     grid position.
/// \param origin[out] Non-null pointer to array of length
///     `component_spec.rank()`.
/// \post `origin[i] == 0` for all unchunked dimensions `i`
/// \post `origin[component_spec.chunked_to_cell_dimensions[j]]` equals
///     `cell_indices[j] * spec.chunk_shape[j]` for all grid dimensions `j`.
void GetComponentOrigin(const ChunkGridSpecification& spec,
                        const ChunkGridSpecification::Component& component_spec,
                        span<const Index> cell_indices, span<Index> origin) {
  assert(spec.rank() == cell_indices.size());
  assert(component_spec.rank() == origin.size());
  std::fill_n(origin.begin(), component_spec.rank(), Index(0));
  for (DimensionIndex chunk_dim_i = 0;
       chunk_dim_i < static_cast<DimensionIndex>(
                         component_spec.chunked_to_cell_dimensions.size());
       ++chunk_dim_i) {
    const DimensionIndex cell_dim_i =
        component_spec.chunked_to_cell_dimensions[chunk_dim_i];
    origin[cell_dim_i] =
        cell_indices[chunk_dim_i] * spec.chunk_shape[chunk_dim_i];
  }
}

/// Returns `true` if all components of `node` have been fully overwritten.
///
/// \param node Non-null pointer to transaction node.
bool IsFullyOverwritten(ChunkCache::TransactionNode& node) {
  auto& entry = GetOwningEntry(node);
  const auto& grid = GetOwningCache(entry).grid();
  const auto& component_specs = grid.components;
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = entry.cell_indices();
  for (size_t component_index = 0, num_components = component_specs.size();
       component_index != num_components; ++component_index) {
    const auto& component_spec = component_specs[component_index];
    origin.resize(component_spec.rank());
    GetComponentOrigin(grid, component_spec, cell_indices, origin);
    if (!node.components()[component_index].write_state.IsFullyOverwritten(
            component_spec, origin)) {
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
///    `PartitionIndexTransformOverRegularGrid` to iterate over the set of grid
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
  std::size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;

  absl::Status operator()(internal::LockCollection& lock_collection) const {
    // No locks need to be held throughout read operation.  A temporary lock is
    // held in the `BeginRead` method below only while copying the shared_ptr to
    // the immutable cached chunk data.
    return absl::OkStatus();
  }

  Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry)->grid(), component_spec,
                       entry->cell_indices(), origin);
    auto read_array = ChunkCache::GetReadComponent(
        AsyncCache::ReadLock<ChunkCache::ReadData>(*entry).data(),
        component_index);
    return component_spec.GetReadNDIterable(std::move(read_array), origin,
                                            std::move(chunk_transform), arena);
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
  std::size_t component_index;
  OpenTransactionNodePtr<ChunkCache::TransactionNode> node;

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
    const auto& component_spec =
        GetOwningCache(entry).grid().components[component_index];
    auto& component = node->components()[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry).grid(), component_spec,
                       entry.cell_indices(), origin);
    SharedArrayView<const void> read_array;
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
    }
    return component.GetReadNDIterable(component_spec, origin,
                                       std::move(read_array), read_generation,
                                       std::move(chunk_transform), arena);
  }
};

/// Shared state used while `Read` is in progress.
struct ReadOperationState : public AtomicReferenceCount<ReadOperationState> {
  using Receiver = AnyFlowReceiver<Status, ReadChunk, IndexTransform<>>;
  struct SharedReceiver : public AtomicReferenceCount<SharedReceiver> {
    Receiver receiver;
  };
  ReadOperationState(Receiver receiver) : shared_receiver(new SharedReceiver) {
    // The receiver is stored in a separate reference-counted object, so that it
    // can outlive `ReadOperationState`.  `ReadOperationState` is destroyed when
    // the last chunk is ready (successfully or with an error), but the
    // `receiver` needs to remain until `promise` is ready, which does not
    // necessarily happen until after the last `ReadOperationState` reference is
    // destroyed.
    shared_receiver->receiver = std::move(receiver);
    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    this->promise = std::move(promise);
    execution::set_starting(this->shared_receiver->receiver,
                            [promise = this->promise] {
                              promise.SetResult(absl::CancelledError(""));
                            });
    std::move(future).ExecuteWhenReady(
        [shared_receiver = this->shared_receiver](ReadyFuture<void> future) {
          auto& result = future.result();
          if (result) {
            execution::set_done(shared_receiver->receiver);
          } else {
            execution::set_error(shared_receiver->receiver, result.status());
          }
          execution::set_stopping(shared_receiver->receiver);
        });
  }
  ~ReadOperationState() { promise.SetReady(); }
  IntrusivePtr<SharedReceiver> shared_receiver;

  /// Tracks errors, cancellation, and completion.
  Promise<void> promise;
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
/// 1. `Write` calls `PartitionIndexTransformOverRegularGrid` to iterate over
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
///    computed by `PartitionIndexTransformOverRegularGrid`.  Along with the
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
  std::size_t component_index;
  OpenTransactionNodePtr<ChunkCache::TransactionNode> node;

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
    const auto component_spec = entry.component_specs()[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry).grid(), component_spec,
                       entry.cell_indices(), origin);
    node->MarkSizeUpdated();
    return node->components()[component_index].BeginWrite(
        component_spec, origin, std::move(chunk_transform), arena);
  }

  Future<const void> operator()(WriteChunk::EndWrite,
                                IndexTransformView<> chunk_transform,
                                NDIterable::IterationLayoutView layout,
                                span<const Index> write_end_position,
                                Arena* arena) const {
    auto& entry = GetOwningEntry(*node);
    const auto& component_spec = entry.component_specs()[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry).grid(), component_spec,
                       entry.cell_indices(), origin);
    const bool modified = node->components()[component_index].EndWrite(
        component_spec, origin, chunk_transform, layout, write_end_position,
        arena);
    if (modified) node->is_modified = modified;
    if (modified && IsFullyOverwritten(*node)) {
      node->SetUnconditional();
    }
    if (modified) return node->transaction()->future();
    return {};
  }
};

}  // namespace

ChunkCache::ChunkCache(ChunkGridSpecification grid, Executor executor)
    : grid_(std::move(grid)), executor_(std::move(executor)) {}

void ChunkCache::Read(
    OpenTransactionPtr transaction, std::size_t component_index,
    IndexTransform<> transform, absl::Time staleness,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  assert(component_index >= 0 && component_index < grid().components.size());
  const auto& component_spec = grid().components[component_index];
  IntrusivePtr<ReadOperationState> state(
      new ReadOperationState(std::move(receiver)));
  auto status = PartitionIndexTransformOverRegularGrid(
      component_spec.chunked_to_cell_dimensions, grid().chunk_shape, transform,
      [&](span<const Index> grid_cell_indices,
          IndexTransformView<> cell_transform) {
        if (!state->promise.result_needed()) {
          return absl::CancelledError("");
        }
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto cell_to_source, ComposeTransforms(transform, cell_transform));
        auto entry = GetEntryForCell(grid_cell_indices);
        // Arrange to call `set_value` on the receiver with a `ReadChunk`
        // corresponding to this grid cell once the read request completes
        // successfully.
        ReadChunk chunk;
        chunk.transform = std::move(cell_to_source);
        Future<const void> read_future;
        if (transaction) {
          TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                                       GetTransactionNode(*entry, transaction));
          read_future = node->IsUnconditional() ? MakeReadyFuture()
                                                : node->Read(staleness);
          chunk.impl =
              ReadChunkTransactionImpl{component_index, std::move(node)};
        } else {
          read_future = entry->Read(staleness);
          chunk.impl = ReadChunkImpl{component_index, std::move(entry)};
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
    state->promise.SetResult(std::move(status));
  }
}

void ChunkCache::Write(
    OpenTransactionPtr transaction, std::size_t component_index,
    IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  ABSL_ASSERT(component_index >= 0 &&
              component_index < grid().components.size());
  // In this implementation, chunks are always available for writing
  // immediately.  The entire stream of chunks is sent to the receiver before
  // this function returns.
  const auto& component_spec = grid().components[component_index];
  std::atomic<bool> cancelled{false};
  execution::set_starting(receiver, [&cancelled] { cancelled = true; });
  Status status = PartitionIndexTransformOverRegularGrid(
      component_spec.chunked_to_cell_dimensions, grid().chunk_shape, transform,
      [&](span<const Index> grid_cell_indices,
          IndexTransformView<> cell_transform) {
        if (cancelled) return absl::CancelledError("");
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto cell_to_dest, ComposeTransforms(transform, cell_transform));
        auto entry = GetEntryForCell(grid_cell_indices);
        auto transaction_copy = transaction;
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto node, GetTransactionNode(*entry, transaction_copy));
        execution::set_value(
            receiver,
            WriteChunk{WriteChunkImpl{component_index, std::move(node)},
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

PinnedCacheEntry<ChunkCache> ChunkCache::GetEntryForCell(
    span<const Index> grid_cell_indices) {
  assert(static_cast<size_t>(grid_cell_indices.size()) ==
         grid().chunk_shape.size());
  const absl::string_view key(
      reinterpret_cast<const char*>(grid_cell_indices.data()),
      grid_cell_indices.size() * sizeof(Index));
  return GetCacheEntry(this, key);
}

void ChunkCache::TransactionNode::Delete() {
  UniqueWriterLock lock(*this);
  this->MarkSizeUpdated();
  this->is_modified = true;
  auto& entry = GetOwningEntry(*this);
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = entry.cell_indices();
  const auto& grid = GetOwningCache(entry).grid();
  for (Index component_index = 0, num_components = grid.components.size();
       component_index != num_components; ++component_index) {
    const auto& component_spec = grid.components[component_index];
    origin.resize(component_spec.rank());
    GetComponentOrigin(grid, component_spec, cell_indices, origin);
    // There is no need to check the reference count of the component data array
    // (i.e. to check for a concurrent read) because this doesn't modify the
    // data array, it just resets the data pointer to `nullptr`.
    components()[component_index].write_state.WriteFillValue(component_spec,
                                                             origin);
  }
  SetUnconditional();
}

Future<const void> ChunkCache::Entry::Delete(OpenTransactionPtr transaction) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                               GetTransactionNode(*this, transaction));
  node->Delete();
  return node->transaction()->future();
}

std::size_t ChunkCache::TransactionNode::ComputeWriteStateSizeInBytes() {
  std::size_t total = 0;
  const auto component_specs = this->component_specs();
  for (size_t component_index = 0;
       component_index < static_cast<size_t>(component_specs.size());
       ++component_index) {
    total +=
        this->components()[component_index].write_state.EstimateSizeInBytes(
            component_specs[component_index]);
  }
  return total;
}

std::size_t ChunkCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  const ReadData* components = static_cast<const ReadData*>(read_data);
  std::size_t total = 0;
  auto component_specs = this->component_specs();
  for (size_t component_index = 0;
       component_index < static_cast<size_t>(component_specs.size());
       ++component_index) {
    total += component_specs[component_index].EstimateReadStateSizeInBytes(
        components[component_index].valid());
  }
  return total;
}

ChunkCache::WritebackSnapshot::WritebackSnapshot(
    TransactionNode& node, AsyncCache::ReadView<ReadData> read_state) {
  auto& entry = GetOwningEntry(node);
  auto& cache = GetOwningCache(entry);
  const auto component_specs = node.component_specs();
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = entry.cell_indices();
  for (std::size_t component_i = 0;
       component_i < static_cast<size_t>(component_specs.size());
       ++component_i) {
    auto& component_spec = component_specs[component_i];
    auto& component = node.components()[component_i];
    origin.resize(component_spec.rank());
    GetComponentOrigin(cache.grid(), component_spec, cell_indices, origin);
    auto component_snapshot = component.GetArrayForWriteback(
        component_spec, origin,
        GetReadComponent(read_state.data(), component_i),
        read_state.stamp().generation);
    if (!component_snapshot.equals_fill_value) {
      if (!new_read_data_) {
        new_read_data_ = internal::make_shared_for_overwrite<ReadData[]>(
            component_specs.size());
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

namespace {
bool IsCommitUnconditional(ChunkCache::TransactionNode& node) {
  return node.IsUnconditional() || !node.is_modified;
}
}  // namespace

void ChunkCache::TransactionNode::DoApply(ApplyOptions options,
                                          ApplyReceiver receiver) {
  if (options.validate_only) {
    execution::set_value(
        receiver, ReadState{{}, TimestampedStorageGeneration::Unconditional()},
        UniqueWriterLock<TransactionNode>{});
    return;
  }
  auto continuation = WithExecutor(
      GetOwningCache(*this).executor_,
      [this, receiver = std::move(receiver)](
          tensorstore::ReadyFuture<const void> future) mutable {
        if (!future.result().ok()) {
          return execution::set_error(receiver, future.result().status());
        }
        AsyncCache::ReadState read_state;
        if (!IsCommitUnconditional(*this)) {
          read_state = AsyncCache::ReadLock<void>(*this).read_state();
        } else {
          read_state.stamp = TimestampedStorageGeneration::Unconditional();
        }
        std::shared_ptr<const void> new_data;
        UniqueWriterLock<AsyncCache::TransactionNode> lock;
        if (is_modified) {
          // Protect against concurrent calls to `DoApply`, since this may
          // modify the write arrays to incorporate the read state.
          lock = UniqueWriterLock<AsyncCache::TransactionNode>(*this);
          WritebackSnapshot snapshot(
              *this, AsyncCache::ReadView<ReadData>(read_state));
          read_state.data = std::move(snapshot.new_read_data());
          read_state.stamp.generation.MarkDirty();
        }
        execution::set_value(receiver, std::move(read_state), std::move(lock));
      });
  if (IsCommitUnconditional(*this)) {
    continuation(MakeReadyFuture());
  } else {
    this->Read(options.staleness_bound)
        .ExecuteWhenReady(std::move(continuation));
  }
}

void ChunkCache::TransactionNode::InvalidateReadState() {
  AsyncCache::TransactionNode::InvalidateReadState();
  for (auto& component : components()) {
    component.InvalidateReadState();
  }
}

ChunkCacheDriver::ChunkCacheDriver(CachePtr<ChunkCache> cache,
                                   std::size_t component_index,
                                   StalenessBound data_staleness_bound)
    : cache_(std::move(cache)),
      component_index_(component_index),
      data_staleness_bound_(data_staleness_bound) {
  assert(cache_);
  assert(component_index < cache_->grid().components.size());
}

DataType ChunkCacheDriver::data_type() {
  return cache_->grid().components[component_index_].data_type();
}

DimensionIndex ChunkCacheDriver::rank() {
  return cache_->grid().components[component_index_].rank();
}

void ChunkCacheDriver::Read(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  cache_->Read(std::move(transaction), component_index_, std::move(transform),
               data_staleness_bound_, std::move(receiver));
}

void ChunkCacheDriver::Write(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  cache_->Write(std::move(transaction), component_index_, std::move(transform),
                std::move(receiver));
}

ChunkCacheDriver::~ChunkCacheDriver() = default;

Executor ChunkCacheDriver::data_copy_executor() { return cache_->executor(); }

}  // namespace internal
}  // namespace tensorstore
