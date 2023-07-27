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

#ifndef TENSORSTORE_INTERNAL_CHUNK_RECEIVER_UTILS_H_
#define TENSORSTORE_INTERNAL_CHUNK_RECEIVER_UTILS_H_

#include "absl/status/status.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

template <typename ChunkT>
struct ChunkOperationState
    : public AtomicReferenceCount<ChunkOperationState<ChunkT>> {
  using ChunkType = ChunkT;
  using BaseReceiver =
      AnyFlowReceiver<absl::Status, ChunkType, IndexTransform<>>;

  struct SharedReceiver : public AtomicReferenceCount<SharedReceiver> {
    BaseReceiver receiver;
  };

  ChunkOperationState(BaseReceiver&& receiver)
      : shared_receiver(new SharedReceiver) {
    // The receiver is stored in a separate reference-counted object, so that it
    // can outlive `ChunkOperationState`.  `ChunkOperationState` is destroyed
    // when the last chunk is ready (successfully or with an error), but the
    // `receiver` needs to remain until `promise` is ready, which does not
    // necessarily happen until after the last `ChunkOperationState` reference
    // is destroyed.
    shared_receiver->receiver = std::move(receiver);
    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    this->promise = std::move(promise);
    execution::set_starting(
        this->shared_receiver->receiver, [promise = this->promise] {
          SetDeferredResult(promise, absl::CancelledError(""));
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
  virtual ~ChunkOperationState() { promise.SetReady(); }

  void SetError(absl::Status&& status) {
    SetDeferredResult(promise, std::move(status));
  }

  bool cancelled() const { return !promise.result_needed(); }

  IntrusivePtr<SharedReceiver> shared_receiver;

  /// Tracks errors, cancellation, and completion.
  Promise<void> promise;
};

// Forwarding receiver which satisfies `ReadChunkReceiver` or
// `WriteChunkReceiver`.  The starting/stopping/error/done parts of the protocol
// are handled by the future, so this only forwards set_error and set_value
// calls.
template <typename StateType>
struct ForwardingChunkOperationReceiver {
  using ChunkType = typename StateType::ChunkType;
  IntrusivePtr<StateType> state;
  IndexTransform<> cell_transform;
  FutureCallbackRegistration cancel_registration;

  void set_starting(AnyCancelReceiver cancel) {
    cancel_registration =
        state->promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration(); }
  void set_done() {}
  void set_error(absl::Status error) { state->SetError(std::move(error)); }
  void set_value(ChunkType chunk, IndexTransform<> composed_transform) {
    auto c_transform = ComposeTransforms(cell_transform, composed_transform);
    if (!c_transform.ok()) {
      state->SetError(std::move(c_transform).status());
    } else {
      execution::set_value(state->shared_receiver->receiver, std::move(chunk),
                           std::move(c_transform).value());
    }
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CHUNK_RECEIVER_UTILS_H_
