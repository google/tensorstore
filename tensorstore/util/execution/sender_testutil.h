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

#ifndef TENSORSTORE_UTIL_SENDER_TESTUTIL_H_
#define TENSORSTORE_UTIL_SENDER_TESTUTIL_H_

#include <stddef.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/synchronization/notification.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sync_flow_sender.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

// Logs the calls to set_value/set_error/etc.
struct LoggingReceiver {
  std::vector<std::string>* log;

  void set_starting(AnyCancelReceiver cancel) {
    log->push_back("set_starting");
  }

  template <typename... V>
  void set_value(V&&... v) {
    if constexpr ((... && internal::IsOstreamable<V>)) {
      log->push_back(tensorstore::StrCat(
          "set_value: ", absl::StrJoin(std::make_tuple(std::forward<V>(v)...),
                                       ", ", absl::StreamFormatter())));
    } else {
      log->push_back(tensorstore::StrCat(
          "set_value: ",
          absl::StrJoin(std::make_tuple(std::forward<V>(v)...), ", ")));
    }
  }

  void set_done() { log->push_back("set_done"); }

  template <typename E>
  void set_error(E&& e) {
    log->push_back(tensorstore::StrCat("set_error: ", e));
  }

  void set_cancel() { log->push_back("set_cancel"); }

  void set_stopping() { log->push_back("set_stopping"); }
};

template <typename Receiver>
struct CompletionNotifyingReceiver
    : public tensorstore::SyncFlowReceiver<Receiver> {
  using Base = tensorstore::SyncFlowReceiver<Receiver>;

  CompletionNotifyingReceiver(absl::Notification* notification,
                              Receiver receiver_arg)
      : Base(std::move(receiver_arg)), notification_(notification) {}

  friend void set_stopping(CompletionNotifyingReceiver& self) {
    tensorstore::execution::set_stopping(static_cast<Base&>(self));
    self.notification_->Notify();
  }
  absl::Notification* notification_;
};

template <typename Receiver>
CompletionNotifyingReceiver(absl::Notification*, Receiver receiver)
    -> CompletionNotifyingReceiver<Receiver>;

// Cancellation on the first call to set_starting() stops the stream.
struct CancelOnStartingReceiver : public LoggingReceiver {
  void set_starting(AnyCancelReceiver do_cancel) {
    this->LoggingReceiver::set_starting({});
    do_cancel();
  }
};

// Cancellation in the middle of the stream stops the stream.
template <size_t N>
struct CancelAfterNReceiver : public LoggingReceiver {
  AnyCancelReceiver cancel;

  void set_starting(AnyCancelReceiver do_cancel) {
    this->cancel = std::move(do_cancel);
    this->LoggingReceiver::set_starting({});
  }

  template <typename... V>
  void set_value(V&&... v) {
    this->LoggingReceiver::set_value(std::forward<V>(v)...);
    if (this->log->size() == N) {
      this->cancel();
    }
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_SENDER_TESTUTIL_H_
