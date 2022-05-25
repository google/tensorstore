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

#include "tensorstore/driver/write.h"

#include <atomic>
#include <memory>
#include <utility>

#include "absl/status/status.h"
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
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {

/// Local state for the asynchronous operation initiated by the `DriverWrite`
/// function.
///
/// `DriverWrite` asynchronously performs the following steps:
///
/// 1. Resolves the `target_transform` bounds from the `DriverWriteTarget`.
///    When completed, control continues with `DriverWriteInitiateOp`.
///
/// 2. Validates that the resolved target bounds match the normalized bounds of
///    the `source` array.
///
/// 3. Calls `Driver::Write` with a `WriteChunkReceiver` to initiate the actual
///    write over the resolved `target_transform` bounds.  `WriteChunkReceiver`
///    ensures that the write is canceled when `copy_promise.result_needed()`
///    becomes `false`.
///
/// 4. For each `WriteChunk` received, `WriteChunkReceiver` invokes
///    `WriteChunkOp` using `executor` to copy the data from the appropriate
///    portion of the `source` array to the `WriteChunk`.
///
/// 5. `WriteChunkOp` links the writeback `Future` returned from the write
///    operation on `WriteChunk` to `copy_promise` with a `CommitCallback` that
///    holds a reference-counted pointer to `CommitState` (but not to
///    `WriteState`).
///
/// 5. Once all `WriteChunkOp` calls have completed (either successfully or with
///    an error), all references to `WriteState` are released, causing it to be
///    destroyed, which in turn releases all references to `copy_promise`, which
///    causes `copy_promise` to become ready.  `CommitCallback` calls may,
///    however, still be outstanding at this point.  Note that `copy_promise` is
///    never marked ready, even with an error, while the `source` array may
///    still be accessed, because the user is permitted to destroy or reuse the
///    `source` array as soon as `copy_promise` becomes ready.
///
/// 6. Once `WriteState` is destroyed and all `CommitCallback` links are
///    completed, the `commit_promise` is marked ready, indicating to the caller
///    that all data has been written back (or an error has occurred).
struct WriteState : public internal::AtomicReferenceCount<WriteState> {
  /// CommitState is a separate reference-counted struct (rather than simply
  /// using `WriteState`) in order to ensure the reference to `copy_promise` and
  /// `source` are released once copying has completed (in order for
  /// `copy_promise` to become ready and for any memory used by `source`, if not
  /// otherwise referenced, to be freed).
  struct CommitState : public internal::AtomicReferenceCount<CommitState> {
    WriteProgressFunction write_progress_function;
    Index total_elements;
    std::atomic<Index> copied_elements{0};
    std::atomic<Index> committed_elements{0};

    void UpdateCopyProgress(Index num_elements) {
      if (!write_progress_function) return;
      write_progress_function(WriteProgress{
          total_elements, copied_elements += num_elements, committed_elements});
    }

    void UpdateCommitProgress(Index num_elements) {
      if (!write_progress_function) return;
      write_progress_function(WriteProgress{
          total_elements, copied_elements, committed_elements += num_elements});
    }
  };
  Executor executor;
  TransformedArray<Shared<const void>> source;
  DataTypeConversionLookupResult data_type_conversion;
  DriverPtr target_driver;
  internal::OpenTransactionPtr target_transaction;
  DomainAlignmentOptions alignment_options;
  Promise<void> copy_promise;
  Promise<void> commit_promise;
  IntrusivePtr<CommitState> commit_state{new CommitState};

  void SetError(absl::Status error) {
    SetDeferredResult(copy_promise, std::move(error));
  }
};

/// Callback invoked by `WriteChunkReceiver` (using the executor) to copy data
/// from the appropriate portion of the source array to a single `WriteChunk`.
struct WriteChunkOp {
  IntrusivePtr<WriteState> state;
  WriteChunk chunk;
  IndexTransform<> cell_transform;
  void operator()() {
    // Map the portion of the source array that corresponds to this chunk
    // to the index space expected by the chunk.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto source,
        ApplyIndexTransform(std::move(cell_transform), state->source),
        state->SetError(_));

    DefaultNDIterableArena arena;

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto source_iterable,
        GetTransformedArrayNDIterable(std::move(source), arena),
        state->SetError(_));

    LockCollection lock_collection;

    absl::Status copy_status;
    Future<const void> commit_future;

    {
      TENSORSTORE_ASSIGN_OR_RETURN(auto guard,
                                   LockChunks(lock_collection, chunk.impl),
                                   state->SetError(_));

      TENSORSTORE_ASSIGN_OR_RETURN(
          auto target_iterable,
          chunk.impl(WriteChunk::BeginWrite{}, chunk.transform, arena),
          state->SetError(_));

      source_iterable = GetConvertedInputNDIterable(
          std::move(source_iterable), target_iterable->dtype(),
          state->data_type_conversion);

      NDIterableCopier copier(*source_iterable, *target_iterable,
                              chunk.transform.input_shape(), arena);
      copy_status = copier.Copy();
      auto end_write_result =
          chunk.impl(WriteChunk::EndWrite{}, chunk.transform,
                     copier.layout_info().layout_view(),
                     copier.stepper().position(), arena);
      commit_future = std::move(end_write_result.commit_future);
      if (copy_status.ok()) {
        copy_status = std::move(end_write_result.copy_status);
      }
    }

    if (copy_status.ok()) {
      const Index num_elements = chunk.transform.input_domain().num_elements();
      state->commit_state->UpdateCopyProgress(num_elements);
      struct CommitCallback {
        IntrusivePtr<WriteState::CommitState> state;
        Index num_elements;
        void operator()(Promise<void>, Future<const void>) const {
          state->UpdateCommitProgress(num_elements);
        }
      };
      if (!state->commit_promise.null() && !commit_future.null()) {
        // For transactional writes, `state->commit_promise` is null.
        LinkValue(CommitCallback{state->commit_state, num_elements},
                  state->commit_promise, std::move(commit_future));
      } else {
        state->commit_state->UpdateCommitProgress(num_elements);
      }
    } else {
      state->SetError(std::move(copy_status));
    }
  }
};

/// FlowReceiver used by `DriverWrite` to copy data from the source array to
/// chunks as they become available.
struct WriteChunkReceiver {
  IntrusivePtr<WriteState> state;
  FutureCallbackRegistration cancel_registration;
  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        state->copy_promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) { state->SetError(std::move(error)); }
  void set_value(WriteChunk chunk, IndexTransform<> cell_transform) {
    // Defer all work to the executor, because we don't know on which thread
    // this may be called.
    //
    // Don't move `state` since `set_value` may be called multiple times.
    state->executor(
        WriteChunkOp{state, std::move(chunk), std::move(cell_transform)});
  }
};

/// Callback used by `DriverWrite` to initiate the write once the target
/// transform bounds have been resolved.
struct DriverWriteInitiateOp {
  IntrusivePtr<WriteState> state;
  void operator()(Promise<void> promise,
                  ReadyFuture<IndexTransform<>> target_transform_future) {
    IndexTransform<> target_transform =
        std::move(target_transform_future.value());
    // Align `source` to the resolved bounds.
    TENSORSTORE_ASSIGN_OR_RETURN(
        state->source.transform(),
        AlignTransformTo(std::move(state->source.transform()),
                         target_transform.domain(), state->alignment_options),
        static_cast<void>(promise.SetResult(_)));
    state->commit_state->total_elements =
        target_transform.domain().num_elements();
    state->copy_promise = std::move(promise);

    // Initiate the write on the driver.
    auto target_driver = std::move(state->target_driver);
    auto target_transaction = std::move(state->target_transaction);
    target_driver->Write(std::move(target_transaction),
                         std::move(target_transform),
                         WriteChunkReceiver{std::move(state)});
  }
};

}  // namespace

WriteFutures DriverWrite(Executor executor,
                         TransformedSharedArray<const void> source,
                         DriverHandle target, DriverWriteOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.driver.read_write_mode()));
  IntrusivePtr<WriteState> state(new WriteState);
  state->executor = executor;
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->data_type_conversion,
      GetDataTypeConverterOrError(source.dtype(), target.driver->dtype(),
                                  options.data_type_conversion_flags));
  state->target_driver = std::move(target.driver);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->target_transaction,
      internal::AcquireOpenTransactionPtrOrError(target.transaction));
  state->source = std::move(source);
  state->alignment_options = options.alignment_options;
  state->commit_state->write_progress_function =
      std::move(options.progress_function);
  auto copy_pair = PromiseFuturePair<void>::Make(MakeResult());
  PromiseFuturePair<void> commit_pair;
  if (!state->target_transaction) {
    commit_pair =
        PromiseFuturePair<void>::LinkError(MakeResult(), copy_pair.future);
    state->commit_promise = std::move(commit_pair.promise);
  } else {
    commit_pair.future = copy_pair.future;
  }

  // Resolve the bounds for `target.transform`.
  auto transform_future = state->target_driver->ResolveBounds(
      state->target_transaction, std::move(target.transform),
      fix_resizable_bounds);

  // Initiate the write once the bounds have been resolved.
  LinkValue(WithExecutor(std::move(executor),
                         DriverWriteInitiateOp{std::move(state)}),
            std::move(copy_pair.promise), std::move(transform_future));
  return {std::move(copy_pair.future), std::move(commit_pair.future)};
}

WriteFutures DriverWrite(TransformedSharedArray<const void> source,
                         DriverHandle target, WriteOptions options) {
  auto executor = target.driver->data_copy_executor();
  return internal::DriverWrite(
      std::move(executor), std::move(source), std::move(target),
      /*options=*/
      {/*.progress_function=*/std::move(options.progress_function),
       /*.alignment_options=*/options.alignment_options});
}

}  // namespace internal
}  // namespace tensorstore
