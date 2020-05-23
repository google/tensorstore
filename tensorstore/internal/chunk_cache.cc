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
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/rank.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/strided_layout.h"
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

/// Returns `true` if all components of `entry` have been fully overwritten.
///
/// \param entry Non-null pointer to entry.
bool IsFullyOverwritten(ChunkCache::Entry* entry) {
  const ChunkCache* cache = GetOwningCache(entry);
  const auto& grid = cache->grid();
  const auto& component_specs = grid.components;
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = entry->cell_indices();
  for (Index component_index = 0,
             num_components = cache->grid().components.size();
       component_index != num_components; ++component_index) {
    const auto& component_spec = component_specs[component_index];
    origin.resize(component_spec.rank());
    GetComponentOrigin(grid, component_spec, cell_indices, origin);
    if (!entry->components[component_index].write_state.IsFullyOverwritten(
            component_spec, origin)) {
      return false;
    }
  }
  return true;
}

/// TensorStore Driver ReadChunk implementation for the chunk cache.
///
/// The `ChunkCache::Read` operation proceeds as follows:
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
///    `AsyncStorageBackedCache::Entry::Read` on the `ChunkCache::Entry` object
///    to wait for data meeting the user-specified `staleness` constraint to be
///    ready.
///
/// 4. If data within the staleness bound is already available for the
///    particular component array of the cell, no actual read operation occurs.
///    Otherwise, this results in a call to the `ChunkCache::DoRead` method
///    which must be overridden by derived classes.
///
/// 5. The derived class implementation of `DoRead` arranges to call
///    `ChunkCache::NotifyReadSuccess` when the read has finished successfully.
///    If there is no data for the cell in the underlying storage, the cell is
///    considered to implicitly contain the fill value, but no storage is
///    actually allocated.
///
/// 6. Once the cell data has been updated (if necessary), the `ReadChunk`
///    constructed previously is sent to the user-specified `receiver`.
struct ReadChunkImpl {
  std::size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;

  Result<NDIterable::Ptr> operator()(ReadChunk::AcquireReadLock,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena) const {
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    auto& component = entry->components[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry)->grid(), component_spec,
                       entry->cell_indices(), origin);
    auto lock = entry->AcquireReadStateReaderLock();
    return component.GetReadNDIterable(component_spec, origin,
                                       std::move(chunk_transform), arena);
  }
  // No-op, since returned `NDIterable` holds a `shared_ptr` to the data.
  void operator()(ReadChunk::ReleaseReadLock) const {}
};

/// Shared state used while `Read` is in progress.
struct ReadState : public AtomicReferenceCount<ReadState> {
  using Receiver = AnyFlowReceiver<Status, ReadChunk, IndexTransform<>>;
  struct SharedReceiver : public AtomicReferenceCount<SharedReceiver> {
    Receiver receiver;
  };
  ReadState(Receiver receiver) : shared_receiver(new SharedReceiver) {
    // The receiver is stored in a separate reference-counted object, so that it
    // can outlive `ReadState`.  `ReadState` is destroyed when the last chunk is
    // ready (successfully or with an error), but the `receiver` needs to remain
    // until `promise` is ready, which does not necessarily happen until after
    // the last `ReadState` reference is destroyed.
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
  ~ReadState() { promise.SetReady(); }
  IntrusivePtr<SharedReceiver> shared_receiver;

  /// Tracks errors, cancellation, and completion.
  Promise<void> promise;
};

/// TensorStore Driver WriteChunk implementation for the chunk cache.
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
///    (creating it if it does not already exist), and immediately
///    (synchronously) sends to the user-specified `receiver` a `WriteChunk`
///    object that holds a pointer to the `ChunkCache::Entry` object, the
///    `component_index`, and a transform from a synthetic "chunk" index space
///    to the index space over which the array component is defined.  (This
///    transform is obtained by simply composing the user-specified `transform`
///    with the `cell_transform` computed by
///    `PartitionIndexTransformOverRegularGrid`.  Along with the `WriteChunk`
///    object, the `cell_transform` is also sent to the `receiver`; the
///    `receiver` uses this `cell_transform` to convert from the domain of the
///    original input domain (i.e. the input domain of the user-specified
///    `transform`) to the synthetic "chunk" index space that must be used to
///    write to the chunk.  Storage for the cell (if not previously allocated)
///    is not actually allocated until the user writes to the chunk.
///
/// 3. Writes made to `WriteChunk` objects sent to the `receiver` result in
///    calls to `WriteToMask`, which provides the necessary tracking to support
///    optimistic (atomic read-modify-write) concurrency.  After the write, the
///    `ChunkCache` implementation calls `AsyncStorageBackedCache::FinishWrite`
///    to obtain a writeback `Future` for the chunk to return to the caller.
///    Writeback is deferred until `Force` is called on the returned `Future`
///    (or a different writeback `Future` obtained for the same cell), or it is
///    triggered automatically by the CachePool due to memory pressure.  If all
///    positions within a given cell of a component array are locally
///    overwritten with not-yet-written-back data, subsequent reads are always
///    satisfied immediately from the cache alone, regardless of the specified
///    staleness bound.
///
/// Phase II: Writeback to persistent storage
/// -----------------------------------------
///
/// 4. When writeback is initiated by the `AsyncStorageBackedCache`
///    implementation, it results in a calls to the `ChunkCache::DoRead` and/or
///    `ChunkCache::DoWriteback` methods which must be overridden by derived
///    classes.
///
/// 5. `DoRead` is called if there is no existing read result for the cell and
///    the cell has not been completely locally overwritten.  The derived class
///    implementation of `DoRead` is responsible for fetching and decoding the
///    existing data for the cell, if available, and calling
///    `ChunkCache::NotifyReadSuccess` with the decoded component arrays for the
///    cell.  The `ChunkCache` implementation merges the updated component
///    arrays (or the fill value arrays if there is no existing data) with the
///    local modifications using `RebaseMaskedArray`.
///
/// 6. Once any necessary reads complete, `DoWriteback` is called.  The
///    implementation of `DoWriteback` uses the RAII `WritebackSnapshot` object
///    to obtain an atomic snapshot of the cell data, which it can then encode
///    appropriately and issue a write request to the underlying storage system.
///    Typically, the write request should be conditional if the write only
///    partially overwrites the cell, and should be unconditional if the write
///    fully overwrites the cell.  A separate `MaskData` data structure tracks
///    writes to the chunk made after the snapshot is taken (i.e. after the
///    `WritebackSnapshot` object is destroyed) but before the writeback
///    completes.
///
/// 7. The derived class `DoWriteback` implementation arranges to call
///    `ChunkCache::NotifyWritebackSuccess` when the write to the underlying
///    storage system completes successfully.  If the writeback fails due to a
///    concurrent modification (i.e. a `StorageGeneration` mismatch), the
///    `DoWriteback` implementation instead calls
///    `ChunkCache::NotifyWritebackNeedsRead` and the `ChunkCache`
///    implementation (facilitated by the `AsyncStorageBackedCache`
///    implementation) returns to step 5 and issues another read request.
///    Otherwise, writeback is considered complete.
struct WriteChunkImpl {
  std::size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;

  // On successful return, implicitly transfers ownership of a `WriteStateLock`
  // to the caller.
  Result<NDIterable::Ptr> operator()(
      WriteChunk::AcquireWriteLock, IndexTransform<> chunk_transform,
      Arena* arena) const ABSL_NO_THREAD_SAFETY_ANALYSIS {
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry)->grid(), component_spec,
                       entry->cell_indices(), origin);
    auto lock = entry->AcquireWriteStateLock();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iterable,
        entry->components[component_index].write_state.BeginWrite(
            component_spec, origin, std::move(chunk_transform), arena));
    // Implicitly transfer ownership of lock to caller.  Note: `release()`
    // does not unlock.
    lock.release();
    return iterable;
  }

  // Unlocks the exclusive lock on `entry->write_mutex` acquired by
  // successful call to the `AcquireWriteLock` method.
  Future<const void> operator()(WriteChunk::ReleaseWriteLock,
                                IndexTransformView<> chunk_transform,
                                NDIterable::IterationLayoutView layout,
                                span<const Index> write_end_position,
                                Arena* arena) const {
    ChunkCache::WriteStateLock lock(entry.get(), std::adopt_lock);
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    absl::FixedArray<Index, kNumInlinedDims> origin(component_spec.rank());
    GetComponentOrigin(GetOwningCache(entry)->grid(), component_spec,
                       entry->cell_indices(), origin);
    const bool modified =
        entry->components[component_index].write_state.EndWrite(
            component_spec, origin, chunk_transform, layout, write_end_position,
            arena);
    if (modified) {
      // The data array was modified.  Notify `AsyncStorageBackedCache` and
      // return the associated writeback `Future`.
      const bool is_fully_overwritten = IsFullyOverwritten(entry.get());

      // Hand off lock to the `FinishWrite` method of
      // `AsyncStorageBackedCache`, which will release it.
      return entry->FinishWrite(
          std::move(lock),
          is_fully_overwritten
              ? AsyncStorageBackedCache::WriteFlags::kUnconditionalWriteback
              : AsyncStorageBackedCache::WriteFlags::kConditionalWriteback);
    }
    entry->AbortWrite(std::move(lock));
    return {};
  }
};

}  // namespace

ChunkCache::ChunkCache(ChunkGridSpecification grid) : grid_(std::move(grid)) {}

void ChunkCache::Read(
    std::size_t component_index, IndexTransform<> transform,
    StalenessBound staleness,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  ABSL_ASSERT(component_index >= 0 &&
              component_index < grid().components.size());
  const auto& component_spec = grid().components[component_index];
  IntrusivePtr<ReadState> state(new ReadState(std::move(receiver)));
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
        auto read_future = entry->Read(staleness);
        // Arrange to call `set_value` on the receiver with a `ReadChunk`
        // corresponding to this grid cell once the read request completes
        // successfully.
        LinkValue(
            [state,
             chunk = ReadChunk{ReadChunkImpl{component_index, std::move(entry)},
                               std::move(cell_to_source)},
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
    std::size_t component_index, IndexTransform<> transform,
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
        execution::set_value(
            receiver,
            WriteChunk{WriteChunkImpl{component_index, std::move(entry)},
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

void ChunkCache::DoInitializeEntry(Cache::Entry* base_entry) {
  auto* entry = static_cast<Entry*>(base_entry);
  entry->components.reserve(grid().components.size());
  for (const auto& component_spec : grid().components) {
    entry->components.emplace_back(component_spec.rank());
  }
}

Future<const void> ChunkCache::Entry::Delete() {
  // Deleting is equivalent to fully overwriting all components with the fill
  // value.
  auto lock = this->AcquireWriteStateLock();
  auto* cache = GetOwningCache(this);
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = this->cell_indices();
  const auto& grid = cache->grid();
  for (Index component_index = 0,
             num_components = cache->grid().components.size();
       component_index != num_components; ++component_index) {
    const auto& component_spec = grid.components[component_index];
    origin.resize(component_spec.rank());
    GetComponentOrigin(grid, component_spec, cell_indices, origin);
    components[component_index].write_state.WriteFillValue(component_spec,
                                                           origin);
  }
  return this->FinishWrite(std::move(lock),
                           WriteFlags::kUnconditionalWriteback);
}

std::size_t ChunkCache::DoGetReadStateSizeInBytes(Cache::Entry* base_entry) {
  std::size_t total = 0;
  Entry* entry = static_cast<Entry*>(base_entry);
  for (Index component_index = 0, num_components = grid_.components.size();
       component_index != num_components; ++component_index) {
    total += entry->components[component_index].EstimateReadStateSizeInBytes(
        grid_.components[component_index]);
  }
  return total;
}

std::size_t ChunkCache::DoGetWriteStateSizeInBytes(Cache::Entry* base_entry) {
  std::size_t total = 0;
  Entry* entry = static_cast<Entry*>(base_entry);
  for (Index component_index = 0, num_components = grid_.components.size();
       component_index != num_components; ++component_index) {
    total += entry->components[component_index].EstimateWriteStateSizeInBytes(
        grid_.components[component_index]);
  }
  return total;
}

ChunkCache::WritebackSnapshot::WritebackSnapshot(Entry* entry) {
  auto lock = entry->AcquireWriteStateLock();
  auto* cache = GetOwningCache(entry);
  const auto& component_specs = cache->grid().components;
  component_arrays_.resize(component_specs.size());
  // Indicates whether all components (processed so far) are equal to the fill
  // value.
  equals_fill_value_ = true;
  unconditional_ = true;
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  const span<const Index> cell_indices = entry->cell_indices();
  for (std::size_t component_i = 0; component_i < component_specs.size();
       ++component_i) {
    auto& component_spec = component_specs[component_i];
    auto& component = entry->components[component_i];
    origin.resize(component_spec.rank());
    GetComponentOrigin(cache->grid(), component_spec, cell_indices, origin);
    auto component_snapshot =
        component.GetArrayForWriteback(component_spec, origin);
    component_arrays_[component_i] = std::move(component_snapshot.array);
    unconditional_ = unconditional_ && component_snapshot.unconditional;
    equals_fill_value_ =
        equals_fill_value_ && component_snapshot.equals_fill_value;
  }
  cache->NotifyWritebackStarted(entry, std::move(lock));
}

void ChunkCache::NotifyReadSuccess(
    Cache::Entry* base_entry, ReadStateWriterLock lock,
    span<const SharedArrayView<const void>> components) {
  auto* entry = static_cast<Entry*>(base_entry);
  const auto& spec = grid();
  const auto& component_specs = spec.components;
  if (!components.empty()) {
    assert(static_cast<size_t>(components.size()) == component_specs.size());
  }
  for (std::size_t component_i = 0; component_i < component_specs.size();
       ++component_i) {
    auto& component = entry->components[component_i];
    if (components.empty()) {
      component.read_array = nullptr;
    } else {
      component.read_array = components[component_i];
    }
  }
  this->NotifyReadSuccess(entry, std::move(lock));
}

namespace {

void AfterWritebackCompletes(ChunkCache* cache, ChunkCache::Entry* entry,
                             bool success) {
  const auto& grid = cache->grid();
  size_t num_components = grid.components.size();
  const span<const Index> cell_indices = entry->cell_indices();
  absl::InlinedVector<Index, kNumInlinedDims> origin;
  for (size_t component_i = 0; component_i < num_components; ++component_i) {
    auto& component = entry->components[component_i];
    const auto& component_spec = grid.components[component_i];
    origin.resize(component_spec.rank());
    GetComponentOrigin(grid, component_spec, cell_indices, origin);
    component.AfterWritebackCompletes(component_spec, origin, success);
  }
}
}  // namespace

void ChunkCache::NotifyWritebackSuccess(Cache::Entry* base_entry,
                                        WriteAndReadStateLock lock) {
  AfterWritebackCompletes(this, static_cast<Entry*>(base_entry),
                          /*success=*/true);
  AsyncStorageBackedCache::NotifyWritebackSuccess(base_entry, std::move(lock));
}

void ChunkCache::NotifyWritebackNeedsRead(Cache::Entry* base_entry,
                                          WriteStateLock lock,
                                          absl::Time staleness_bound) {
  AfterWritebackCompletes(this, static_cast<Entry*>(base_entry),
                          /*success=*/false);
  AsyncStorageBackedCache::NotifyWritebackNeedsRead(base_entry, std::move(lock),
                                                    staleness_bound);
}

void ChunkCache::NotifyWritebackError(Cache::Entry* base_entry,
                                      WriteStateLock lock, Status error) {
  AfterWritebackCompletes(this, static_cast<Entry*>(base_entry),
                          /*success=*/false);
  AsyncStorageBackedCache::NotifyWritebackError(base_entry, std::move(lock),
                                                std::move(error));
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
    IndexTransform<> transform,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  cache_->Read(component_index_, std::move(transform), data_staleness_bound_,
               std::move(receiver));
}

void ChunkCacheDriver::Write(
    IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  cache_->Write(component_index_, std::move(transform), std::move(receiver));
}

ChunkCacheDriver::~ChunkCacheDriver() = default;

}  // namespace internal
}  // namespace tensorstore
