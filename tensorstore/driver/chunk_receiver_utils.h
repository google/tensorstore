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

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/flow_sender_operation_state.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

// Forwarding receiver which satisfies `ReadChunkReceiver` or
// `WriteChunkReceiver`.  The starting/stopping/error/done parts of the protocol
// are handled by the future, so this only forwards set_error and set_value
// calls.
template <typename ChunkType, typename StateType>
struct ForwardingChunkOperationReceiver {
  IntrusivePtr<StateType> state;
  IndexTransform<> cell_transform;
  FutureCallbackRegistration cancel_registration;

  // StateType must be a FlowSenderOperationState or a subclass of that.
  static_assert(
      std::is_same_v<FlowSenderOperationState<ChunkType, IndexTransform<>>,
                     StateType> ||
      std::is_base_of_v<FlowSenderOperationState<ChunkType, IndexTransform<>>,
                        StateType>);

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
