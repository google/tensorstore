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

#include "tensorstore/driver/read.h"

#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {

/// Local state for the asynchronous operation initiated by the two `DriverRead`
/// overloads.
///
/// `DriverRead` asynchronously performs the following steps:
///
/// 1. Resolves the bounds in `source` via `Driver::ResolveBounds`.  When
///    completed, control continues with `DriverReadIntoExistingInitiateOp` (if
///    using an existing `target` array) or `DriverReadIntoNewInitiateOp` (if
///    using a new `target` array).
///
/// 2. If reading into an existing `target` array, validates that the resolved
///    source bounds match the normalized bounds of the `target` array.
///    Otherwise, allocates a new `target` array with a domain given by the
///    source bounds.
///
/// 3. Calls `Driver::Read` with a `ReadChunkReceiver` to initiate the actual
///    read over the resolved `source.transform` bounds.  `ReadChunkReceiver`
///    ensures that the read is canceled if `promise.result_needed()` becomes
///    `false`.
///
/// 4. For each `ReadChunk` received, `ReadChunkReceiver` invokes `ReadChunkOp`
///    using `executor` to copy the data from the `ReadChunk` to the appropriate
///    portion of the `target` array.
///
/// 5. Once all work has finished (either because all chunks were processed
///    successfully, an error occurred, or all references to the future
///    associated with `promise` were released), all references to `ReadState`
///    are released, which in turn releases all references to `promise`, which
///    causes `promise` to become ready.  Note that `promise` is never marked
///    ready, even with an error, while the `target` array may still be
///    accessed, because the user is permitted to destroy or reuse the `target`
///    array as soon as the promise becomes ready.
template <typename PromiseValue>
struct ReadState
    : public internal::AtomicReferenceCount<ReadState<PromiseValue>> {
  Executor executor;
  DriverPtr source_driver;
  internal::OpenTransactionPtr source_transaction;
  DataTypeConversionLookupResult data_type_conversion;
  TransformedArray<Shared<void>> target;
  DomainAlignmentOptions alignment_options;
  ReadProgressFunction read_progress_function;
  Promise<PromiseValue> promise;
  std::atomic<Index> copied_elements{0};
  Index total_elements;

  void SetError(absl::Status error) {
    SetDeferredResult(promise, std::move(error));
  }

  void UpdateProgress(Index num_elements) {
    if (!read_progress_function) return;
    read_progress_function(
        ReadProgress{total_elements, copied_elements += num_elements});
  }
};

/// Callback invoked by `ReadChunkReceiver` (using the executor) to copy data
/// from a single `ReadChunk` to the appropriate portion of the `target` array.
template <typename PromiseValue>
struct ReadChunkOp {
  IntrusivePtr<ReadState<PromiseValue>> state;
  ReadChunk chunk;
  IndexTransform<> cell_transform;
  void operator()() {
    // Map the portion of the target array that corresponds to this chunk to
    // the index space expected by the chunk.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto target,
        ApplyIndexTransform(std::move(cell_transform), state->target),
        state->SetError(_));
    absl::Status copy_status =
        internal::CopyReadChunk(chunk.impl, std::move(chunk.transform),
                                state->data_type_conversion, target);
    if (copy_status.ok()) {
      state->UpdateProgress(ProductOfExtents(target.shape()));
    } else {
      state->SetError(std::move(copy_status));
    }
  }
};

/// FlowReceiver used by the two `DriverRead` overloads to copy data from chunks
/// as they become available.
template <typename PromiseValue>
struct ReadChunkReceiver {
  IntrusivePtr<ReadState<PromiseValue>> state;
  FutureCallbackRegistration cancel_registration;
  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        state->promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) { state->SetError(std::move(error)); }
  void set_value(ReadChunk chunk, IndexTransform<> cell_transform) {
    // Defer all work to the executor, because we don't know on which thread
    // this may be called.
    state->executor(ReadChunkOp<PromiseValue>{state, std::move(chunk),
                                              std::move(cell_transform)});
  }
};

/// Callback used by `DriverRead` to initiate a read into an existing array once
/// the source transform bounds have been resolved.
struct DriverReadIntoExistingInitiateOp {
  using State = ReadState<void>;
  IntrusivePtr<State> state;
  void operator()(Promise<void> promise,
                  ReadyFuture<IndexTransform<>> source_transform_future) {
    IndexTransform<> source_transform =
        std::move(source_transform_future.value());
    // Align the resolved bounds to `target`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        source_transform,
        AlignTransformTo(std::move(source_transform), state->target.domain(),
                         state->alignment_options),
        static_cast<void>(promise.SetResult(_)));
    state->promise = std::move(promise);
    state->total_elements = source_transform.domain().num_elements();

    // Initiate the read on the driver.
    auto source_driver = std::move(state->source_driver);
    auto source_transaction = std::move(state->source_transaction);
    source_driver->Read(std::move(source_transaction),
                        std::move(source_transform),
                        ReadChunkReceiver<void>{std::move(state)});
  }
};

/// Callback used by `DriverRead` to initiate a read into a new array once the
/// source transform bounds have been resolved.
struct DriverReadIntoNewInitiateOp {
  using State = ReadState<SharedOffsetArray<void>>;
  IntrusivePtr<State> state;
  DataType target_dtype;
  ContiguousLayoutOrder target_layout_order;
  void operator()(Promise<SharedOffsetArray<void>> promise,
                  ReadyFuture<IndexTransform<>> source_transform_future) {
    IndexTransform<> source_transform =
        std::move(source_transform_future.value());

    if (!IsFinite(source_transform.domain())) {
      promise.SetResult(absl::InvalidArgumentError(tensorstore::StrCat(
          "Read requires a finite domain, got ", source_transform.domain())));
      return;
    }

    auto array = AllocateArray(source_transform.domain().box(),
                               target_layout_order, default_init, target_dtype);
    auto& r = promise.raw_result() = std::move(array);
    state->target = *r;
    state->promise = std::move(promise);
    state->total_elements = source_transform.input_domain().num_elements();

    // Initiate the read on the driver.
    auto source_driver = std::move(state->source_driver);
    auto source_transaction = std::move(state->source_transaction);
    source_driver->Read(
        std::move(source_transaction), std::move(source_transform),
        ReadChunkReceiver<SharedOffsetArray<void>>{std::move(state)});
  }
};

}  // namespace

Future<void> DriverRead(Executor executor, DriverHandle source,
                        TransformedSharedArray<void> target,
                        DriverReadOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.driver.read_write_mode()));
  using State = ReadState<void>;
  IntrusivePtr<State> state(new State);
  state->executor = executor;
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->data_type_conversion,
      GetDataTypeConverterOrError(source.driver->dtype(), target.dtype(),
                                  options.data_type_conversion_flags));
  state->source_driver = std::move(source.driver);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->source_transaction,
      internal::AcquireOpenTransactionPtrOrError(source.transaction));
  state->target = std::move(target);
  state->alignment_options = options.alignment_options;
  state->read_progress_function = std::move(options.progress_function);
  auto pair = PromiseFuturePair<void>::Make(MakeResult());

  // Resolve the bounds for `source.transform`.
  auto transform_future = state->source_driver->ResolveBounds(
      state->source_transaction, std::move(source.transform),
      fix_resizable_bounds);

  // Initiate the read once the bounds have been resolved.
  LinkValue(WithExecutor(std::move(executor),
                         DriverReadIntoExistingInitiateOp{std::move(state)}),
            std::move(pair.promise), std::move(transform_future));
  return std::move(pair.future);
}

Future<void> DriverRead(DriverHandle source,
                        TransformedSharedArray<void> target,
                        ReadOptions options) {
  auto executor = source.driver->data_copy_executor();
  return internal::DriverRead(
      std::move(executor), std::move(source), std::move(target), /*options=*/
      {/*.progress_function=*/std::move(options.progress_function),
       /*.alignment_options=*/options.alignment_options});
}

Future<SharedOffsetArray<void>> DriverReadIntoNewArray(
    Executor executor, DriverHandle source, DataType target_dtype,
    ContiguousLayoutOrder target_layout_order,
    DriverReadIntoNewOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.driver.read_write_mode()));
  using State = ReadState<SharedOffsetArray<void>>;
  IntrusivePtr<State> state(new State);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->data_type_conversion,
      GetDataTypeConverterOrError(source.driver->dtype(), target_dtype));
  state->executor = executor;
  state->source_driver = std::move(source.driver);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->source_transaction,
      internal::AcquireOpenTransactionPtrOrError(source.transaction));
  state->read_progress_function = std::move(options.progress_function);
  auto pair = PromiseFuturePair<SharedOffsetArray<void>>::Make();

  // Resolve the bounds for `source.transform`.
  auto transform_future = state->source_driver->ResolveBounds(
      state->source_transaction, std::move(source.transform),
      fix_resizable_bounds);

  // Initiate the read once the bounds have been resolved.
  LinkValue(
      WithExecutor(std::move(executor),
                   DriverReadIntoNewInitiateOp{std::move(state), target_dtype,
                                               target_layout_order}),
      std::move(pair.promise), std::move(transform_future));
  return std::move(pair.future);
}

Future<SharedOffsetArray<void>> DriverReadIntoNewArray(
    DriverHandle source, ReadIntoNewArrayOptions options) {
  auto dtype = source.driver->dtype();
  auto executor = source.driver->data_copy_executor();
  return internal::DriverReadIntoNewArray(
      std::move(executor), std::move(source), dtype, options.layout_order,
      /*options=*/
      {/*.progress_function=*/std::move(options.progress_function)});
}

absl::Status CopyReadChunk(
    ReadChunk::Impl& chunk, IndexTransform<> chunk_transform,
    const DataTypeConversionLookupResult& chunk_conversion,
    TransformedArray<void, dynamic_rank, view> target) {
  DefaultNDIterableArena arena;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto target_iterable,
      GetTransformedArrayNDIterable(UnownedToShared(target), arena));

  LockCollection lock_collection;
  TENSORSTORE_ASSIGN_OR_RETURN(auto guard, LockChunks(lock_collection, chunk));

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto source_iterable,
      chunk(ReadChunk::BeginRead{}, std::move(chunk_transform), arena));

  source_iterable = GetConvertedInputNDIterable(
      std::move(source_iterable), target_iterable->dtype(), chunk_conversion);

  // Copy the chunk to the relevant portion of the target array.
  NDIterableCopier copier(*source_iterable, *target_iterable, target.shape(),
                          arena);
  return copier.Copy();
}

absl::Status CopyReadChunk(ReadChunk::Impl& chunk,
                           IndexTransform<> chunk_transform,
                           TransformedArray<void, dynamic_rank, view> target) {
  auto converter =
      internal::GetDataTypeConverter(target.dtype(), target.dtype());
  return CopyReadChunk(chunk, std::move(chunk_transform), converter,
                       std::move(target));
}

}  // namespace internal
}  // namespace tensorstore
