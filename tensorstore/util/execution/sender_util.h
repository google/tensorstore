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

#ifndef TENSORSTORE_UTIL_EXECUTION_SENDER_UTIL_H_
#define TENSORSTORE_UTIL_EXECUTION_SENDER_UTIL_H_

#include <atomic>
#include <iterator>
#include <tuple>
#include <utility>

#include "tensorstore/util/execution/execution.h"

namespace tensorstore {


/// Receiver that adapts a FlowReceiver to be used as a single Receiver.
template <typename FlowReceiver>
struct FlowSingleReceiver {
  FlowReceiver receiver;

  template <typename... V>
  void set_value(V... v) {
    execution::set_starting(receiver, [] {});
    execution::set_value(receiver, std::move(v)...);
    execution::set_done(receiver);
    execution::set_stopping(receiver);
  }

  template <typename E>
  void set_error(E e) {
    execution::set_starting(receiver, [] {});
    execution::set_error(receiver, std::move(e));
    execution::set_stopping(receiver);
  }

  void set_cancel() {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
  }
};
template <typename FlowReceiver>
FlowSingleReceiver(FlowReceiver receiver) -> FlowSingleReceiver<FlowReceiver>;

/// FlowSender that adapts a single Sender to be used as FlowSender.
template <typename Sender>
struct FlowSingleSender {
  Sender sender;
  template <typename Receiver>
  void submit(Receiver receiver) {
    execution::submit(sender,
                      FlowSingleReceiver<Receiver>{std::move(receiver)});
  }
};
template <typename Sender>
FlowSingleSender(Sender sender) -> FlowSingleSender<Sender>;

/// FlowSender that consecutively sends each element of a range.
///
/// \tparam Range Type compatible with a range-based for loop.
template <typename Range>
struct RangeFlowSender {
  Range range;
  template <typename Receiver>
  friend void submit(RangeFlowSender& sender, Receiver receiver) {
    std::atomic<bool> cancelled{false};
    execution::set_starting(receiver, [&cancelled] { cancelled = true; });
    using std::begin;
    using std::end;
    auto it = begin(sender.range);
    auto end_it = end(sender.range);
    for (; !cancelled && it != end_it; ++it) {
      auto&& value = *it;
      execution::set_value(receiver, std::forward<decltype(value)>(value));
    }
    execution::set_done(receiver);
    execution::set_stopping(receiver);
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_SENDER_UTIL_H_
