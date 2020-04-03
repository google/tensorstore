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
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
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

ChunkGridSpecification::Component::Component(
    SharedArray<const void> fill_value) {
  this->fill_value = fill_value;
  chunked_to_cell_dimensions.resize(fill_value.rank());
  std::iota(chunked_to_cell_dimensions.begin(),
            chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));
}

ChunkGridSpecification::Component::Component(
    SharedArray<const void> fill_value,
    std::vector<DimensionIndex> chunked_to_cell_dimensions)
    : fill_value(std::move(fill_value)),
      chunked_to_cell_dimensions(std::move(chunked_to_cell_dimensions)) {}

ChunkGridSpecification::ChunkGridSpecification(Components components_arg)
    : components(std::move(components_arg)) {
  assert(components.size() > 0);
  // Extract the chunk shape from the cell shape of the first component.
  chunk_shape.resize(components[0].chunked_to_cell_dimensions.size());
  for (DimensionIndex i = 0;
       i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
    chunk_shape[i] =
        components[0].cell_shape()[components[0].chunked_to_cell_dimensions[i]];
  }
  // Verify that the extents of the chunked dimensions are the same for all
  // components.
#if !defined(NDEBUG)
  for (const auto& component : components) {
    assert(component.chunked_to_cell_dimensions.size() == chunk_shape.size());
    for (DimensionIndex i = 0;
         i < static_cast<DimensionIndex>(chunk_shape.size()); ++i) {
      assert(chunk_shape[i] ==
             component.cell_shape()[component.chunked_to_cell_dimensions[i]]);
    }
  }
#endif  // !defined(NDEBUG)
}

namespace {

/// Computes the bounding box of a cell for a particular component array at the
/// specified grid position.
///
/// \param spec Grid specification.
/// \param component_spec Component specification.
/// \param cell_indices Pointer to array of length `spec.rank()` specifying the
///     grid position.
/// \param box[out] Non-null pointer where the result is stored.
/// \post `RangesEqual(box->shape(), component_spec.cell_shape())`
/// \post `box->origin()[i] == 0` for all unchunked dimensions `i`
/// \post `box->origin()[component_spec.chunked_to_cell_dimensions[j]]` equals
///     `cell_indices[j] * spec.chunk_shape[j]` for all grid dimensions `j`.
void GetComponentBox(const ChunkGridSpecification& spec,
                     const ChunkGridSpecification::Component& component_spec,
                     const Index* cell_indices,
                     Box<dynamic_rank(kNumInlinedDims)>* box) {
  *box = BoxView(component_spec.cell_shape());
  span<Index> origin = box->origin();
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

/// Ensures the data array for a given component has been allocated.
///
/// \param component_spec Component specification.
/// \param component[out] Non-null pointer to component.
/// \returns `true` if a new data array was allocated, `false` if the data array
///     was previously allocated.
bool EnsureDataAllocated(
    const ChunkGridSpecification::Component& component_spec,
    ChunkCache::Entry::Component* component) {
  if (!component->data) {
    component->data = AllocateAndConstructSharedElements(
                          component_spec.fill_value.num_elements(),
                          default_init, component_spec.fill_value.data_type())
                          .pointer();
    return true;
  }
  return false;
}

/// Returns `true` if all components of `entry` have been fully overwritten.
///
/// \param entry Non-null pointer to entry.
bool IsFullyOverwritten(ChunkCache::Entry* entry) {
  const ChunkCache* cache = GetOwningCache(entry);
  const auto& component_specs = cache->grid().components;
  for (std::size_t component_i = 0; component_i < component_specs.size();
       ++component_i) {
    const auto& component = entry->components[component_i];
    const auto& component_spec = component_specs[component_i];
    if (component.write_mask.num_masked_elements !=
        component_spec.fill_value.num_elements()) {
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
///    particular component array of the cell, or the cell's component array has
///    been completely overwritten locally (in which case any externally-updated
///    data is irrelevant), no actual read operation occurs.  Otherwise, this
///    results in a call to the `ChunkCache::DoRead` method which must be
///    overridden by derived classes.
///
/// 5. The derived class implementation of `DoRead` arranges to call
///    `ChunkCache::ReadReceiver::NotifyDone` when the read has finished and
///    either an error has occurred or the cell data has been decoded
///    successfully.  If there is no data for the cell in the underlying storage
///    (as indicated by an error code of `absl::StatusCode::kNotFound`), the
///    cell is considered to implicitly contain the fill value, but no storage
///    is actually allocated.  If updated data for the cell was decoded, it is
///    merged with any local writes to the cell using `RebaseMaskedArray`.
///
/// 6. Once the cell data has been updated (if necessary), the `ReadChunk`
///    constructed previously is sent to the user-specified `receiver`.
struct ReadChunkImpl {
  std::size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;

  // On successful return, implicitly transfers ownership of a shared lock on
  // `entry->data_mutex` to the caller.
  Result<NDIterable::Ptr> operator()(
      ReadChunk::AcquireReadLock, IndexTransform<> chunk_transform,
      Arena* arena) const ABSL_NO_THREAD_SAFETY_ANALYSIS {
    // Hold shared reader lock on the `data_mutex` until `ReleaseReadLock` is
    // called.
    UniqueReaderLock read_lock(entry->data_mutex);
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    auto& component = entry->components[component_index];
    Box<dynamic_rank(kNumInlinedDims)> box;
    GetComponentBox(GetOwningCache(entry)->grid(), component_spec,
                    entry->cell_indices().data(), &box);
    StridedLayoutView<dynamic_rank, offset_origin> data_layout;
    absl::FixedArray<Index, kNumInlinedDims> byte_strides(
        component_spec.rank());
    ElementPointer<const void> element_pointer;
    if (!component.data) {
      // No data array has been allocated for the component cell.  Since the
      // read request completed successfully, that means the component cell
      // implicitly contains the fill value.
      data_layout = StridedLayoutView<dynamic_rank, offset_origin>{
          box, component_spec.fill_value.byte_strides()};
      element_pointer = component_spec.fill_value.element_pointer();
    } else {
      ComputeStrides(ContiguousLayoutOrder::c,
                     component_spec.fill_value.data_type()->size, box.shape(),
                     byte_strides);
      data_layout =
          StridedLayoutView<dynamic_rank, offset_origin>{box, byte_strides};
      ElementPointer<void> component_element_pointer = {
          component.data, component_spec.fill_value.data_type()};
      element_pointer = component_element_pointer;
      if (!component.valid_outside_write_mask) {
        // A data array has been allocated, but has so far only been used for
        // writes.  Initialize unwritten elements with the fill value.
        //
        // It is safe for `RebaseMaskedArray` to update the unmasked elements
        // of the array while only holding a read lock, because the only
        // possible concurrent access is another thread running the same
        // `RebaseMaskedArray` call.
        RebaseMaskedArray(box, component_spec.fill_value,
                          component_element_pointer, component.write_mask);
        component.valid_outside_write_mask = true;
      }
    }
    element_pointer =
        AddByteOffset(element_pointer, -data_layout.origin_byte_offset());
    TENSORSTORE_ASSIGN_OR_RETURN(
        chunk_transform,
        ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iterable,
        GetNormalizedTransformedArrayNDIterable(
            {element_pointer, std::move(chunk_transform)}, arena));
    // Implicitly transfer ownership of shared lock to caller.  Note:
    // `release()` does not unlock.
    read_lock.release();
    return iterable;
  }
  // Unlocks the shared lock on `entry->data_mutex` acquired by successful call
  // to the `AcquireReadLock` method.
  void operator()(ReadChunk::ReleaseReadLock) const
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    entry->data_mutex.ReaderUnlock();
  }
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
///    `ChunkCache::ReadReceiver::NotifyDone` with the decoded component arrays
///    for the cell.  The `ChunkCache` implementation merges the updated
///    component arrays (or the fill value arrays if there is no existing data)
///    with the local modifications using `RebaseMaskedArray`.
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
///    `ChunkCache::WritebackReceiver::NotifyDone` when the write to the
///    underlying storage system completes (successfully or unsuccessfully).  If
///    the write fails due to a concurrent modification (i.e. a
///    `StorageGeneration` mismatch), as indicated by
///    `StorageGeneration::Unknown()`, the `ChunkCache` implementation
///    (facilitated by the `AsyncStorageBackedCache` implementation) returns to
///    step 5 and issues another read request.  Otherwise, writeback is
///    considered complete (even if an error occurs).  Any retry logic is the
///    responsibility of the derived class implementation of `DoWriteback`.
struct WriteChunkImpl {
  std::size_t component_index;
  PinnedCacheEntry<ChunkCache> entry;

  // On successful return, implicitly transfers ownership of an exclusive lock
  // on `entry->data_mutex` to the caller.
  Result<NDIterable::Ptr> operator()(
      WriteChunk::AcquireWriteLock, IndexTransform<> chunk_transform,
      Arena* arena) const ABSL_NO_THREAD_SAFETY_ANALYSIS {
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    Future<const void> writeback_future;
    Status write_status;
    Box<dynamic_rank(kNumInlinedDims)> box;
    GetComponentBox(GetOwningCache(entry)->grid(), component_spec,
                    entry->cell_indices().data(), &box);
    absl::FixedArray<Index, kNumInlinedDims> byte_strides(
        component_spec.rank());
    ComputeStrides(ContiguousLayoutOrder::c,
                   component_spec.fill_value.data_type()->size, box.shape(),
                   byte_strides);
    StridedLayoutView<dynamic_rank, offset_origin> data_layout{box,
                                                               byte_strides};

    // Hold an exclusive lock on `data_mutex` while allocating the data array
    // (if necessary) and writing to it.
    std::unique_lock<Mutex> lock(entry->data_mutex);
    auto& component = entry->components[component_index];
    const bool allocated_data = EnsureDataAllocated(component_spec, &component);
    ElementPointer<void> data_ptr(component.data,
                                  component_spec.fill_value.data_type());

    if (allocated_data && component.write_mask.num_masked_elements ==
                              component_spec.fill_value.num_elements()) {
      // Previously, there was no data array allocated for the component but
      // it was considered to have been implicitly overwritten with the fill
      // value.  Now that the data array has been allocated, it must actually
      // be initialized with the fill value.
      CopyArray(component_spec.fill_value,
                ArrayView<void>(
                    data_ptr, StridedLayoutView<>(box.shape(), byte_strides)));
      component.valid_outside_write_mask = true;
    }

    data_ptr = AddByteOffset(data_ptr, -data_layout.origin_byte_offset());
    TENSORSTORE_ASSIGN_OR_RETURN(
        chunk_transform,
        ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto iterable, GetNormalizedTransformedArrayNDIterable(
                           {data_ptr, std::move(chunk_transform)}, arena));
    // Implicitly transfer ownership of lock to caller.  Note: `release()` does
    // not unlock.
    lock.release();
    return iterable;
  }

  // Unlocks the exclusive lock on `entry->data_mutex` acquired by successful
  // call to the `AcquireWriteLock` method.
  Future<const void> operator()(WriteChunk::ReleaseWriteLock,
                                IndexTransformView<> chunk_transform,
                                NDIterable::IterationLayoutView layout,
                                span<const Index> write_end_position,
                                Arena* arena) const {
    std::unique_lock<Mutex> lock(entry->data_mutex, std::adopt_lock);
    const auto& component_spec =
        GetOwningCache(entry)->grid().components[component_index];
    Future<const void> writeback_future;
    Box<dynamic_rank(kNumInlinedDims)> box;
    GetComponentBox(GetOwningCache(entry)->grid(), component_spec,
                    entry->cell_indices().data(), &box);
    auto& component = entry->components[component_index];

    const bool modified =
        WriteToMask(&component.write_mask, box, chunk_transform, layout,
                    write_end_position, arena);
    if (modified) {
      // The data array was modified.  Notify `AsyncStorageBackedCache` and
      // return the associated writeback `Future`.
      const bool is_fully_overwritten = IsFullyOverwritten(entry.get());
      const std::size_t new_size =
          GetOwningCache(entry)->DoGetSizeInBytes(entry.get());
      // Hand off lock to the `FinishWrite` method of
      // `AsyncStorageBackedCache`, which will release it.
      return entry->FinishWrite(
          {std::move(lock), new_size},
          is_fully_overwritten
              ? AsyncStorageBackedCache::WriteFlags::kSupersedesRead
              : AsyncStorageBackedCache::WriteFlags::kConditionalWriteback);
    }
    // If `modified == false`, `lock` is implicitly released at the end of this
    // scope.
    return {};
  }
};

}  // namespace

ChunkCache::Entry::Component::Component(DimensionIndex rank)
    : write_mask(rank), write_mask_prior_to_writeback(rank) {}

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

Cache::Entry* ChunkCache::DoAllocateEntry() {
  std::unique_ptr<Entry> entry(new Entry);
  entry->components.reserve(grid().components.size());
  for (const auto& component_spec : grid().components) {
    entry->components.emplace_back(component_spec.rank());
  }
  return entry.release();
}

void ChunkCache::DoDeleteEntry(Cache::Entry* base_entry) {
  Entry* entry = static_cast<Entry*>(base_entry);
  delete entry;
}

void ChunkCache::ReadReceiver::NotifyDone(
    Result<ComponentsWithGeneration> components) const {
  if (!components) {
    // The read failed.
    receiver_.NotifyDone(/*size_update=*/{}, components.status());
    return;
  }
  if (StorageGeneration::IsUnknown(components->generation.generation)) {
    // The existing data was already up to date.
    assert(components->components.empty());
    receiver_.NotifyDone(/*size_update=*/{}, std::move(components->generation));
    return;
  }
  auto& entry = *this->entry();
  std::unique_lock<Mutex> lock(entry.data_mutex);
  auto* cache = GetOwningCache(&entry);
  Box<dynamic_rank(kNumInlinedDims)> box;
  auto* cell_indices = entry.cell_indices().data();
  const auto& spec = cache->grid();
  const auto& component_specs = spec.components;
  if (!components->components.empty()) {
    assert(static_cast<size_t>(components->components.size()) ==
           component_specs.size());
  }
  // Copy/update the data for each component array.  Use the fill value arrays
  // if `components` is not specified (which can only occur at this point in the
  // function due to an error of `absl::StatusCode::kNotFound`).
  const bool use_fill_value = components->components.empty();
  for (std::size_t component_i = 0; component_i < component_specs.size();
       ++component_i) {
    auto& component = entry.components[component_i];
    auto& component_spec = component_specs[component_i];
    if (  // The chunk is no longer present in the underlying storage.
        use_fill_value &&
        (  // Either there are no local modifications to the component,
            component.write_mask.num_masked_elements == 0 ||
            // or the only local modification was to overwrite the chunk with
            // the fill value.
            !component.data)) {
      // The only valid mask states are fully masked (meaning fully overwritten
      // with the fill value, i.e. deleted), or fully unmasked.
      assert(component.write_mask.num_masked_elements == 0 ||
             (component.write_mask.num_masked_elements ==
              component_spec.fill_value.num_elements()));
      // Free any existing mask or data arrays, such that the cell is considered
      // to implicitly contain the fill value.  The mask array, if present, is
      // unnecessary for representing the fully masked or fully unmasked state.
      component.write_mask.mask_array.reset();
      component.data = nullptr;
      component.valid_outside_write_mask = false;
      continue;
    }
    // Otherwise, actually initialize/update the component array data.
    EnsureDataAllocated(component_spec, &component);
    const ArrayView<const void> source_arr =
        use_fill_value ? component_spec.fill_value
                       : components->components[component_i];
    assert(internal::RangesEqual(source_arr.shape(),
                                 component_spec.fill_value.shape()));
    // Update the unmasked portion of the data array with the new data or with
    // the fill value.
    GetComponentBox(spec, component_spec, cell_indices, &box);
    RebaseMaskedArray(
        box, source_arr,
        ElementPointer<void>(component.data,
                             component_spec.fill_value.data_type()),
        component.write_mask);
    component.valid_outside_write_mask = true;
  }
  // Notify the `AsyncStorageBackedCache` layer that the read operation has
  // completed.
  receiver_.NotifyDone({std::move(lock), cache->DoGetSizeInBytes(&entry)},
                       std::move(components->generation));
}

void ChunkCache::WritebackReceiver::NotifyDone(
    Result<TimestampedStorageGeneration> generation) const {
  auto& entry = *this->entry();
  auto* cache = GetOwningCache(&entry);
  std::unique_lock<Mutex> lock(entry.data_mutex);
  if (generation && !StorageGeneration::IsUnknown(generation->generation)) {
    // Writeback was successful.  Reset the prior mask.
    for (auto& component : entry.components) {
      component.write_mask_prior_to_writeback.Reset();
    }
  } else {
    // Writeback failed or was aborted.  For each component array, combine the
    // current mask into the prior mask.
    Box<dynamic_rank(kNumInlinedDims)> box;
    auto* cell_indices = entry.cell_indices().data();
    const auto& spec = cache->grid();
    const auto& component_specs = spec.components;
    for (std::size_t component_i = 0; component_i < component_specs.size();
         ++component_i) {
      auto& component = entry.components[component_i];
      auto& component_spec = component_specs[component_i];
      GetComponentBox(spec, component_spec, cell_indices, &box);
      UnionMasks(box, &component.write_mask,
                 &component.write_mask_prior_to_writeback);
      component.write_mask_prior_to_writeback.Reset();
    }
  }
  // Notify the `AsyncStorageBackedCache` layer that the writeback operation has
  // completed.
  receiver_.NotifyDone({std::move(lock), cache->DoGetSizeInBytes(&entry)},
                       std::move(generation));
}

Future<const void> ChunkCache::Entry::Delete() {
  // Deleting is equivalent to fully overwriting all components with the fill
  // value.
  std::unique_lock<Mutex> lock(data_mutex);
  auto* cache = GetOwningCache(this);
  for (Index component_index = 0,
             num_components = cache->grid().components.size();
       component_index != num_components; ++component_index) {
    auto& component = components[component_index];
    auto& component_spec = cache->grid().components[component_index];
    component.data = nullptr;
    component.valid_outside_write_mask = false;
    component.write_mask.Reset();
    component.write_mask.num_masked_elements =
        component_spec.fill_value.num_elements();
  }
  const std::size_t new_size = cache->DoGetSizeInBytes(this);
  return this->FinishWrite({std::move(lock), new_size},
                           WriteFlags::kSupersedesRead);
}

std::size_t ChunkCache::DoGetSizeInBytes(Cache::Entry* base_entry) {
  std::size_t total =
      AsyncStorageBackedCache::DoGetSizeInBytes(base_entry) + sizeof(Entry);
  Entry* entry = static_cast<Entry*>(base_entry);
  for (Index component_index = 0, num_components = grid_.components.size();
       component_index != num_components; ++component_index) {
    const auto& component = entry->components[component_index];
    const auto& component_spec = grid_.components[component_index];
    const Index num_elements = ProductOfExtents(component_spec.cell_shape());
    if (component.data) {
      total += num_elements * component_spec.fill_value.data_type()->size;
    }
    if (component.write_mask.mask_array) {
      total += num_elements * sizeof(bool);
    }
    if (component.write_mask_prior_to_writeback.mask_array) {
      total += num_elements * sizeof(bool);
    }
  }
  return total;
}

ChunkCache::WritebackSnapshot::WritebackSnapshot(
    const WritebackReceiver& receiver)
    : receiver_(receiver) {
  auto* entry = receiver.entry();
  std::unique_lock<Mutex> lock(entry->data_mutex);
  DimensionIndex total_cell_dims = 0;
  auto* cache = GetOwningCache(entry);
  const auto& component_specs = cache->grid().components;
  // Allocate sufficient memory in a single vector to store the byte strides for
  // all component arrays.
  for (const auto& component_spec : component_specs) {
    total_cell_dims += component_spec.rank();
  }
  byte_strides_.resize(total_cell_dims);
  component_arrays_.resize(component_specs.size());
  total_cell_dims = 0;
  // Indicates whether all components (processed so far) are equal to the fill
  // value.
  equals_fill_value_ = true;
  for (std::size_t component_i = 0; component_i < component_specs.size();
       ++component_i) {
    auto& component_spec = component_specs[component_i];
    auto& component = entry->components[component_i];
    total_cell_dims += component_spec.rank();
    if (!component.data) {
      component_arrays_[component_i] = component_spec.fill_value;
      continue;
    }
    // Construct an ArrayView for this component array.
    const span<Index> byte_strides(byte_strides_.data() + total_cell_dims,
                                   component_spec.rank());
    const DataType data_type = component_spec.fill_value.data_type();
    ComputeStrides(ContiguousLayoutOrder::c, data_type->size,
                   component_spec.cell_shape(), byte_strides);
    ElementPointer<void> component_element_pointer(component.data, data_type);
    ArrayView<void> component_array(
        component_element_pointer,
        StridedLayoutView<>(component_spec.cell_shape(), byte_strides));
    if (!component.valid_outside_write_mask) {
      // The unmasked portion of this component array has not yet been
      // initialized, and implicitly is considered to contain the fill value.
      // For writeback, we need to actually initialize it with the fill value.
      Box<dynamic_rank(kNumInlinedDims)> box;
      GetComponentBox(cache->grid(), component_spec,
                      entry->cell_indices().data(), &box);
      RebaseMaskedArray(box, component_spec.fill_value,
                        component_element_pointer, component.write_mask);
      component.valid_outside_write_mask = true;
    }
    component_arrays_[component_i] = component_array;
    // Check if this component array cell is equal to the fill value.
    equals_fill_value_ =
        equals_fill_value_ && (component_array == component_spec.fill_value);
  }
  // Ensure the `data_mutex` remains locked even after the `lock` object is
  // destroyed.  The `WritebackSnapshot` destructor releases the lock.
  lock.release();
}

ChunkCache::WritebackSnapshot::~WritebackSnapshot() {
  auto* entry = receiver_.entry();
#if !defined(NDEBUG)
  entry->data_mutex.AssertHeld();
#endif
  if (equals_fill_value_) {
    // All component arrays are equal to their fill value.  To save memory,
    // represent them implicitly rather than explicitly.
    for (auto& component : entry->components) {
      component.data = nullptr;
      component.valid_outside_write_mask = false;
    }
  }

  for (auto& component : entry->components) {
    // Save the current `write_mask` in `write_mask_prior_to_writeback`, and
    // reset `write_mask` so that new writes are recorded separately.  If
    // writeback fails, `write_mask_prior_to_writeback` will be combined into
    // `write_mask`.  If writeback succeeds, it will be discarded.
    std::swap(component.write_mask, component.write_mask_prior_to_writeback);
    component.write_mask.Reset();
  }

  const std::size_t new_size = GetOwningCache(entry)->DoGetSizeInBytes(entry);

  receiver_.receiver_.NotifyStarted(
      {std::unique_lock<Mutex>(entry->data_mutex, std::adopt_lock), new_size});
}

void ChunkCache::DoRead(ReadOptions options,
                        AsyncStorageBackedCache::ReadReceiver receiver) {
  this->DoRead(std::move(options), ReadReceiver{std::move(receiver)});
}

void ChunkCache::DoWriteback(
    TimestampedStorageGeneration existing_generation,
    AsyncStorageBackedCache::WritebackReceiver receiver) {
  this->DoWriteback(existing_generation,
                    WritebackReceiver{std::move(receiver)});
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
