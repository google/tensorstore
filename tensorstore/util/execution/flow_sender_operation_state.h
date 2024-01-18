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

#ifndef TENSORSTORE_UTIL_EXECUTION_FLOW_SENDER_OPERATION_STATE_H_
#define TENSORSTORE_UTIL_EXECUTION_FLOW_SENDER_OPERATION_STATE_H_

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

// Base class for asynchronous operation state objects that manages cancellation
// and error returns with an `AnyFlowReceiver<absl::Status, T...>`.
template <typename... T>
struct FlowSenderOperationState
    : public AtomicReferenceCount<FlowSenderOperationState<T...>> {
  using BaseReceiver = AnyFlowReceiver<absl::Status, T...>;

  struct SharedReceiver : public AtomicReferenceCount<SharedReceiver> {
    BaseReceiver receiver;
  };

  explicit FlowSenderOperationState(BaseReceiver&& receiver)
      : shared_receiver(new SharedReceiver) {
    // The receiver is stored in a separate reference-counted object, so that it
    // can outlive `FlowSenderOperationState`.  `FlowSenderOperationState` is
    // destroyed when the asynchronous operation completes (successfully or with
    // an error), but the `receiver` needs to remain until `promise` is ready,
    // which does not necessarily happen until after the last
    // `FlowSenderOperationState` reference is destroyed.
    shared_receiver->receiver = std::move(receiver);

    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    this->promise = std::move(promise);
    execution::set_starting(
        this->shared_receiver->receiver, [promise = this->promise] {
          SetDeferredResult(promise, absl::CancelledError(""));
        });
    future.Force();
    std::move(future).ExecuteWhenReady(
        [shared_receiver = this->shared_receiver](ReadyFuture<void> future) {
          auto& result = future.result();
          if (result.ok() || absl::IsCancelled(result.status())) {
            execution::set_done(shared_receiver->receiver);
          } else {
            execution::set_error(shared_receiver->receiver, result.status());
          }
          execution::set_stopping(shared_receiver->receiver);
        });
  }
  virtual ~FlowSenderOperationState() { promise.SetReady(); }

  void SetError(absl::Status status) {
    SetDeferredResult(promise, std::move(status));
  }

  bool cancelled() const { return !promise.result_needed(); }

  IntrusivePtr<SharedReceiver> shared_receiver;

  /// Tracks errors, cancellation, and completion.
  Promise<void> promise;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_FLOW_SENDER_OPERATION_STATE_H_
