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

#ifndef TENSORSTORE_UTIL_EXECUTION_COLLECTING_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_COLLECTING_SENDER_H_

#include <utility>

#include "tensorstore/util/execution/execution.h"

namespace tensorstore {
namespace internal {

/// CollectingReceiver is a `FlowReceiver` that collects received elements in
/// a container and sends the container to a SingleReceiver when the stream is
/// done.
///
/// The sender to which this receiver is submitted must not make multiple
/// concurrent calls to `set_value`, `set_error`, or `set_done`.  To use this
/// receiver with such a sender, wrap the `CollectingReceiver` in a
/// `SyncFlowReceiver`.
///
/// \tparam Container Container type that supports `emplace_back(V...)`, where
///     `V...` are the value types sent by the sender to which this receiver is
///     submitted.
/// \tparam SingleReceiver Model of `Receiver<E, Container>`, where `E` is the
///     error type of the sender to which this receiver is submitted.
template <typename Container, typename SingleReceiver>
struct CollectingReceiver {
  SingleReceiver receiver;
  Container container;

  template <typename CancelReceiver>
  friend void set_starting(CollectingReceiver& self, CancelReceiver cancel) {
    // Cancellation not supported.
  }

  template <typename... V>
  friend void set_value(CollectingReceiver& self, V... v) {
    self.container.emplace_back(std::move(v)...);
  }

  template <typename E>
  friend void set_error(CollectingReceiver& self, E e) {
    execution::set_error(self.receiver, std::move(e));
  }

  friend void set_done(CollectingReceiver& self) {
    execution::set_value(self.receiver, std::move(self.container));
  }

  friend void set_stopping(CollectingReceiver& self) {}
};

/// Adapts a FlowSender into a SingleSender by collecting received values into
/// the specified `Container` type.
template <typename Container, typename Sender>
struct CollectingSender {
  Sender sender;

  template <typename Receiver>
  friend void submit(CollectingSender& self, Receiver receiver) {
    execution::submit(self.sender, CollectingReceiver<Container, Receiver>{
                                       std::move(receiver)});
  }
};

template <typename Container, typename Sender>
CollectingSender<Container, Sender> MakeCollectingSender(Sender sender) {
  return {std::move(sender)};
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_COLLECTING_SENDER_H_
