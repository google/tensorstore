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

#include "tensorstore/driver/copy.h"

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
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {

/// Local state for the asynchronous operation initiated by `DriverCopy`.
///
/// `DriverCopy` asynchronously performs the following steps:
///
/// 1. Resolves the `source_transform` bounds from the `DriverReadSource` and
///    the `target_transform` bounds from the `DriverWriteTarget`.  When
///    completed, control continues with `DriverCopyInitiateOp`.
///
/// 2. Validates that the resolved source and target bounds match.
///
/// 3. Calls `Driver::Read` on the `source_driver` with a
///    `CopyReadChunkReceiver` to initiate the actual read over the resolved
///    `source_transform` bounds.  `CopyReadChunkReceiver` ensures that the read
///    is canceled if `copy_promise.result_needed()` becomes `false`.
///
/// 4. For each `ReadChunk` received, `CopyReadChunkReceiver` invokes
///    `CopyInitiateWriteOp` using `executor`: `CopyInitiateWriteOp` calls
///    `Driver::Write` on `target_driver` with a `CopyWriteChunkReceiver` to
///    initiate the write to the portion of `target_driver` corresponding to the
///    `ReadChunk`.  `CopyWriteChunkReceiver` ensures that the write is canceled
///    if `copy_promise.result_needed()` becomes `false`.
///
/// 5. For each `WriteChunk` received, `CopyWriteChunkReceiver` invokes
///    `CopyChunkOp` using executor to copy the data from the appropriate
///    portion of the `ReadChunk` to the `WriteChunk`.
///
/// 6. `CopyChunkOp` links the writeback `Future` returned from the write
///    operation on `WriteChunk` to `copy_promise` with a `CommitCallback` that
///    holds a reference-counted pointer to `CommitState` (but not to
///    `WriteState`).
///
/// 7. Once all `CopyChunkOp` calls have completed (either successfully or with
///    an error), all references to `CopyState` are released, causing it to be
///    destroyed, which in turn releases all references to `copy_promise`, which
///    causes `copy_promise` to become ready.  `CommitCallback` calls may,
///    however, still be outstanding at this point.  Note that `copy_promise` is
///    never marked ready, even with an error, while the `source` array may
///    still be accessed, because the user is permitted to destroy or reuse the
///    `source` array as soon as `copy_promise` becomes ready.
///
/// 8. Once `CopyState` is destroyed and all `CommitCallback` links are
///    completed, the `commit_promise` is marked ready, indicating to the caller
///    that all data has been written back (or an error has occurred).
struct CopyState : public internal::AtomicReferenceCount<CopyState> {
  /// CommitState is a separate reference-counted struct (rather than simply
  /// using `CopyState`) in order to ensure the reference to `copy_promise` and
  /// `source` are released once copying has completed (in order for
  /// `copy_promise` to become ready and for `source`, if not otherwise
  /// referenced, to be freed).
  struct CommitState : public internal::AtomicReferenceCount<CommitState> {
    CopyProgressFunction progress_function;
    Index total_elements;
    std::atomic<Index> copied_elements{0};
    std::atomic<Index> committed_elements{0};
    std::atomic<Index> read_elements{0};

    void UpdateReadProgress(Index num_elements) {
      if (!progress_function) return;
      progress_function(CopyProgress{total_elements,
                                     read_elements += num_elements,
                                     copied_elements, committed_elements});
    }

    void UpdateCopyProgress(Index num_elements) {
      if (!progress_function) return;
      progress_function(CopyProgress{total_elements, read_elements,
                                     copied_elements += num_elements,
                                     committed_elements});
    }

    void UpdateCommitProgress(Index num_elements) {
      if (!progress_function) return;
      progress_function(CopyProgress{total_elements, read_elements,
                                     copied_elements,
                                     committed_elements += num_elements});
    }
  };
  Executor executor;
  DriverPtr source_driver;
  internal::OpenTransactionPtr source_transaction;
  DataTypeConversionLookupResult data_type_conversion;
  DriverPtr target_driver;
  internal::OpenTransactionPtr target_transaction;
  IndexTransform<> target_transform;
  DomainAlignmentOptions alignment_options;
  Promise<void> copy_promise;
  Promise<void> commit_promise;
  IntrusivePtr<CommitState> commit_state{new CommitState};

  void SetError(absl::Status error) {
    SetDeferredResult(copy_promise, std::move(error));
  }
};

/// Callback invoked by `CopyWriteChunkReceiver` (using the executor) to copy
/// data from the relevant portion of a single `ReadChunk` to a `WriteChunk`.
struct CopyChunkOp {
  IntrusivePtr<CopyState> state;
  ReadChunk adjusted_read_chunk;
  WriteChunk write_chunk;
  void operator()() {
    DefaultNDIterableArena arena;

    LockCollection lock_collection;

    absl::Status copy_status;
    Future<const void> commit_future;
    {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto guard,
          LockChunks(lock_collection, adjusted_read_chunk.impl,
                     write_chunk.impl),
          state->SetError(_));

      TENSORSTORE_ASSIGN_OR_RETURN(
          auto source_iterable,
          adjusted_read_chunk.impl(ReadChunk::BeginRead{},
                                   std::move(adjusted_read_chunk.transform),
                                   arena),
          state->SetError(_));

      TENSORSTORE_ASSIGN_OR_RETURN(
          auto target_iterable,
          write_chunk.impl(WriteChunk::BeginWrite{}, write_chunk.transform,
                           arena),
          state->SetError(_));

      source_iterable = GetConvertedInputNDIterable(
          std::move(source_iterable), target_iterable->dtype(),
          state->data_type_conversion);

      NDIterableCopier copier(*source_iterable, *target_iterable,
                              write_chunk.transform.input_shape(), arena);
      copy_status = copier.Copy();

      auto end_write_result =
          write_chunk.impl(WriteChunk::EndWrite{}, write_chunk.transform,
                           copier.layout_info().layout_view(),
                           copier.stepper().position(), arena);
      commit_future = std::move(end_write_result.commit_future);
      if (copy_status.ok()) {
        copy_status = std::move(end_write_result.copy_status);
      }
    }
    if (copy_status.ok()) {
      const Index num_elements = write_chunk.transform.domain().num_elements();
      state->commit_state->UpdateCopyProgress(num_elements);
      struct CommitCallback {
        IntrusivePtr<CopyState::CommitState> state;
        Index num_elements;
        void operator()(Promise<void>, Future<const void>) const {
          state->UpdateCommitProgress(num_elements);
        }
      };
      if (!state->commit_promise.null() && !commit_future.null()) {
        // For transactional writes, `state->commit_promise` is null.
        LinkValue(CommitCallback{state->commit_state, num_elements},
                  state->commit_promise, commit_future);
      } else {
        state->commit_state->UpdateCommitProgress(num_elements);
      }
    } else {
      state->SetError(std::move(copy_status));
    }
  }
};

/// FlowReceiver used by `CopyReadChunkReceiver` to copy data from a given
/// `read_chunk` to the target `write_chunk` chunks as the target chunks become
/// available.
struct CopyWriteChunkReceiver {
  IntrusivePtr<CopyState> state;
  ReadChunk read_chunk;
  FutureCallbackRegistration cancel_registration;
  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        state->copy_promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) { state->SetError(std::move(error)); }
  void set_value(WriteChunk write_chunk, IndexTransform<> cell_transform) {
    // Map the portion of the `read_chunk` that corresponds to this
    // `write_chunk` to the index space expected by `write_chunk`, and produce
    // an `adjusted_read_chunk`.
    //
    // We do this immediately, rather than deferring it to an executor, in order
    // to avoid having to make an extra copy of `read_chunk.transform`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto read_chunk_transform,
        ComposeTransforms(read_chunk.transform, std::move(cell_transform)),
        state->SetError(_));
    ReadChunk adjusted_read_chunk{read_chunk.impl,
                                  std::move(read_chunk_transform)};
    // Defer the actual copying to the executor.
    //
    // Don't move `state` since `set_value` may be called multiple times.
    state->executor(CopyChunkOp{state, std::move(adjusted_read_chunk),
                                std::move(write_chunk)});
  }
};

/// Callback invoked by `CopyReadChunkReceiver` (using the executor) to initiate
/// the `Driver::Write` operation on the `target` driver corresponding to a
/// single `ReadChunk` from the `source` driver.
struct CopyInitiateWriteOp {
  IntrusivePtr<CopyState> state;
  ReadChunk chunk;
  IndexTransform<> cell_transform;
  void operator()() {
    // Map the portion of the target TensorStore corresponding to this source
    // `chunk` to the index space expected by `chunk`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto write_transform,
        ComposeTransforms(state->target_transform, cell_transform),
        state->SetError(_));
    state->commit_state->UpdateReadProgress(
        cell_transform.input_domain().num_elements());

    // Initiate a write for the portion of the target TensorStore
    // corresponding to this source `chunk`.
    state->target_driver->Write(
        state->target_transaction, std::move(write_transform),
        CopyWriteChunkReceiver{state, std::move(chunk)});
  }
};

/// FlowReceiver used by `DriverCopy` that receives source chunks as they become
/// available for reading, and initiates a write on the target driver for each
/// chunk received.
struct CopyReadChunkReceiver {
  IntrusivePtr<CopyState> state;
  FutureCallbackRegistration cancel_registration;
  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        state->copy_promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) { state->SetError(std::move(error)); }
  void set_value(ReadChunk chunk, IndexTransform<> cell_transform) {
    // Defer actual work to executor.
    //
    // Don't move `state` since `set_value` may be called multiple times.
    state->executor(CopyInitiateWriteOp{state, std::move(chunk),
                                        std::move(cell_transform)});
  }
};

/// Callback used by `DriverCopy` to initiate the copy operation once the bounds
/// for the source and target transforms have been resolved.
struct DriverCopyInitiateOp {
  IntrusivePtr<CopyState> state;
  void operator()(Promise<void> promise,
                  ReadyFuture<IndexTransform<>> source_transform_future,
                  ReadyFuture<IndexTransform<>> target_transform_future) {
    IndexTransform<> source_transform =
        std::move(source_transform_future.value());
    IndexTransform<> target_transform =
        std::move(target_transform_future.value());
    // Align the resolved `source_transform` domain to the resolved
    // `target_transform` domain.
    TENSORSTORE_ASSIGN_OR_RETURN(
        source_transform,
        AlignTransformTo(std::move(source_transform), target_transform.domain(),
                         state->alignment_options),
        static_cast<void>(promise.SetResult(_)));
    state->commit_state->total_elements =
        target_transform.input_domain().num_elements();
    state->copy_promise = std::move(promise);
    state->target_transform = std::move(target_transform);

    // Initiate the read operation on the source driver.
    auto source_driver = std::move(state->source_driver);
    auto source_transaction = std::move(state->source_transaction);
    source_driver->Read(std::move(source_transaction),
                        std::move(source_transform),
                        CopyReadChunkReceiver{std::move(state)});
  }
};

}  // namespace

WriteFutures DriverCopy(Executor executor, DriverHandle source,
                        DriverHandle target, DriverCopyOptions options) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.driver.read_write_mode()));
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.driver.read_write_mode()));
  IntrusivePtr<CopyState> state(new CopyState);
  state->executor = executor;
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->data_type_conversion,
      GetDataTypeConverterOrError(source.driver->dtype(),
                                  target.driver->dtype(),
                                  options.data_type_conversion_flags));
  state->source_driver = std::move(source.driver);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->source_transaction,
      internal::AcquireOpenTransactionPtrOrError(source.transaction));
  state->target_driver = std::move(target.driver);
  TENSORSTORE_ASSIGN_OR_RETURN(
      state->target_transaction,
      internal::AcquireOpenTransactionPtrOrError(target.transaction));
  state->alignment_options = options.alignment_options;
  state->commit_state->progress_function = std::move(options.progress_function);
  auto copy_pair = PromiseFuturePair<void>::Make(MakeResult());
  PromiseFuturePair<void> commit_pair;
  if (!state->target_transaction) {
    commit_pair =
        PromiseFuturePair<void>::LinkError(MakeResult(), copy_pair.future);
    state->commit_promise = std::move(commit_pair.promise);
  } else {
    commit_pair.future = copy_pair.future;
  }

  // Resolve the source and target bounds.
  auto source_transform_future = state->source_driver->ResolveBounds(
      state->source_transaction, std::move(source.transform),
      fix_resizable_bounds);
  auto target_transform_future = state->target_driver->ResolveBounds(
      state->target_transaction, std::move(target.transform),
      fix_resizable_bounds);

  // Initiate the copy once the bounds have been resolved.
  LinkValue(
      WithExecutor(std::move(executor), DriverCopyInitiateOp{std::move(state)}),
      std::move(copy_pair.promise), std::move(source_transform_future),
      std::move(target_transform_future));
  return {std::move(copy_pair.future), std::move(commit_pair.future)};
}

WriteFutures DriverCopy(DriverHandle source, DriverHandle target,
                        CopyOptions options) {
  auto executor = source.driver->data_copy_executor();
  return internal::DriverCopy(
      std::move(executor), std::move(source), std::move(target),
      /*options=*/
      {/*.progress_function=*/std::move(options.progress_function),
       /*.alignment_options=*/options.alignment_options});
}

}  // namespace internal
}  // namespace tensorstore
