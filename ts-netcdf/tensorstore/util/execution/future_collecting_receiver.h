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

#ifndef TENSORSTORE_UTIL_EXECUTION_FUTURE_COLLECTING_RECEIVER_H_
#define TENSORSTORE_UTIL_EXECUTION_FUTURE_COLLECTING_RECEIVER_H_

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"

namespace tensorstore {

/// Receiver that collects a flow sender into a `Future<Container>`.
///
/// Unlike `CollectingReceiver`, cancellation (via the
/// `Promise::ExecuteWhenNotNeeded` signal) is supported.
template <typename Container>
struct FutureCollectingReceiver {
  Promise<Container> promise;
  Container container;
  FutureCallbackRegistration cancel_registration;

  template <typename... V>
  void set_value(V&&... v) {
    container.emplace_back(std::forward<V>(v)...);
  }

  void set_error(absl::Status status) { promise.SetResult(std::move(status)); }

  void set_done() { promise.SetResult(std::move(container)); }

  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration = promise.ExecuteWhenNotNeeded(std::move(cancel));
  }

  void set_stopping() { cancel_registration.Unregister(); }
};

template <typename Container, typename Sender>
Future<Container> CollectFlowSenderIntoFuture(Sender sender) {
  auto [promise, future] = PromiseFuturePair<Container>::Make();
  execution::submit(std::move(sender),
                    FutureCollectingReceiver<Container>{std::move(promise)});
  return std::move(future);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_FUTURE_COLLECTING_RECEIVER_H_
