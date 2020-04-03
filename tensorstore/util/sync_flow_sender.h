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

#ifndef TENSORSTORE_UTIL_SYNC_FLOW_SENDER_H_
#define TENSORSTORE_UTIL_SYNC_FLOW_SENDER_H_

#include <mutex>  // NOLINT
#include <utility>

#include "tensorstore/util/execution.h"

namespace tensorstore {

/// `SyncFlowReceiver` adapts a FlowReceiver and ensures calls to `set_value`
/// are serialized.
///
/// \tparam Mutex The mutex type to use, must be compatible with
///     `std::lock_guard`.
/// \tparam Receiver The FlowReceiver type to adapt.
///
/// \remark SyncFlowReceiver is movable even though the underlying Mutex is not
///     moved, because the caller must ensure that any moves are serialized with
///     calls to any of the other functions.
template <typename Mutex, typename Receiver>
struct SyncFlowReceiver {
  SyncFlowReceiver(Receiver receiver) : receiver(std::move(receiver)) {}
  SyncFlowReceiver(SyncFlowReceiver&& other)
      : receiver(std::move(other.receiver)) {}

  SyncFlowReceiver& operator=(SyncFlowReceiver&& other) {
    receiver = std::move(other.receiver);
    return *this;
  }

  template <typename CancelReceiver>
  friend void set_starting(SyncFlowReceiver& self, CancelReceiver cancel) {
    // No need for additional serialization because the sender is required to
    // call this prior to calling any other functions.
    execution::set_starting(self.receiver, std::move(cancel));
  }

  template <typename... V>
  friend void set_value(SyncFlowReceiver& self, V... v) {
    std::lock_guard<Mutex> lock(self.mutex);
    execution::set_value(self.receiver, std::move(v)...);
  }

  friend void set_done(SyncFlowReceiver& self) {
    // No need for additional serialization because the sender is required to
    // call this after all other calls except set_stopping, and before
    // set_stopping.
    execution::set_done(self.receiver);
  }

  template <typename E>
  friend void set_error(SyncFlowReceiver& self, E e) {
    // No need for additional serialization because the sender is required to
    // call this after all other calls except set_stopping, and before
    // set_stopping.
    execution::set_error(self.receiver, std::move(e));
  }

  friend void set_stopping(SyncFlowReceiver& self) {
    // No need for additional serialization because the sender is required to
    // call this after all other calls.
    execution::set_stopping(self.receiver);
  }

  Mutex mutex;
  Receiver receiver;
};

/// FlowSender that adapts a FlowSender to ensure calls to the receiver
/// functions are serialized.
template <typename Mutex, typename Sender>
struct SyncFlowSender {
  Sender sender;

  template <typename Receiver>
  friend void submit(SyncFlowSender& self, Receiver receiver) {
    execution::submit(self.sender,
                      SyncFlowReceiver<Mutex, Receiver>(std::move(receiver)));
  }
};

template <typename Mutex = std::mutex, typename Sender>
SyncFlowSender<Mutex, Sender> MakeSyncFlowSender(Sender sender) {
  return {std::move(sender)};
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_SYNC_FLOW_SENDER_H_
