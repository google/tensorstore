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

#ifndef TENSORSTORE_UTIL_SENDER_H_
#define TENSORSTORE_UTIL_SENDER_H_

/// \file
/// Asynchronous channel interfaces.
///
/// A model of the `Sender<E, V...>` concept represents an asynchronous
/// producer/channel of a single tuple of values of types `V...`, which may
/// signal an error of type `E` in lieu of producing a value.  The counterpart
/// to a `Sender<E, V...>` is a `Receiver<E, V...>`, which represents a
/// continuation that either receives a tuple of values of types `V...` or an
/// error value of type `E`.  The usage is as follows:
///
///   1. A receiver is bound to a sender by invoking
///      `tensorstore::execution::submit(sender, receiver)`.  If `sender`
///      represents a deferred computation, the actual computation may not begin
///      until `submit` is called..
///
///   2. The sender sends a value by calling
///      `tensorstore::execution::set_value(receiver, v...)`, sends an error by
///      calling `tensorstore::execution::set_error(receiver, e)`, or indicates
///      that the computation was cancelled by calling
///      `tensorstore::execution::set_cancel(receiver)`.  At most one of these
///      functions may be called.
///
/// Unless explicitly permitted by a particular Sender, invoking `submit` more
/// than once results in undefined behavior.  Likewise, unless explicitly
/// permitted by a particular Receiver, invoking any of `set_value`, `set_error`
/// or `set_cancel` after any one of those functions has already been called
/// results in undefined behavior.
///
/// The `Sender` concept does not itself provide an in-band way for the receiver
/// to signal cancellation, but cancellation could be signaled through some
/// separate mechanism (e.g. a `Sender<E>` used by the consumer to signal
/// cancellation that is provided to the sender when it is created).
///
/// A model of the related `FlowSender<E, V...>` concept represents a producer
/// of a channel/stream of values that supports in-band cancellation (in
/// contrast, the `Sender`/`Receiver` concepts support transmission of just a
/// single result).  The consumer end of the channel is represented by a model
/// of the `FlowReceiver<E, V...>` concept:
///
///   1. A receiver is bound to a sender by invoking
///      `tensorstore::execution::submit(sender, receiver)`.
///
///   2. The sender calls
///      `tensorstore::execution::set_starting(receiver, cancel)`, where
///      `cancel` is a nullary function that may be called by the receiver to
///      request cancellation.  The provided `cancel` function is only valid to
///      call until the receiver's `set_stopping` callback has returned.
///
///   3. The sender calls `tensorstore::execution::set_value(receiver, v...)`
///      zero or more times to provide a stream of value tuples.
///
///   4. The sender indicates an error by calling
///      `tensorstore::execution::set_error(receiver, e)`, or indicates success
///      by calling `tensorstore::execution::set_done(receiver)`.  At most one
///      of `set_error` or `set_done` may be called, and `set_value` must not be
///      called after `set_error` or `set_done` is called.
///
///   5. The sender calls `tensorstore::execution::set_stopping(receiver)` to
///      indicate that the `cancel` function provided to `set_starting` is no
///      longer valid.
///
/// The `submit`, `set_starting`, `set_value`, `set_done`,`set_error`,
/// `set_cancel`, and `set_stopping` functions defined in the
/// `tensorstore::execution` namespace are function objects that serve as the
/// interface to customization points: for each customization point `X`,
/// `tensorstore::execution::X(obj, args...)` invokes `obj.X(args...)` if that
/// is defined, or otherwise invokes `X(obj, args...)` where `X` is found
/// through argument-dependent lookup (ADL).  To make a new or existing type
/// model one of the above mentioned concepts, simply define the appropriate
/// customization points as methods or free functions.
///
/// This file defines the above mentioned customization points as well as
/// type-erasure classes `AnySender`, `AnyReceiver`, `AnyFlowSender`, and
/// `AnyFlowReceiver`.
///
/// Refer to the definitions of `DoneSender`, `ErrorSender`, `ValueSender`,
/// `SenderWithExecutor`, `FlowSingleSender` and `RangeFlowSender` below, and
/// `LoggingReceiver` in `sender_testutil.h`, as examples.
///
///
///
/// The `tensorstore::Result<T>` type models both `Sender<Status, T>` and
/// `Receiver<Status, T>`.  The `tensorstore::Future<T>` type models
/// `Sender<Status, T>`, while `tensorstore::Promise<T>` models
/// `Receiver<Status, T>`.  The `MakeSenderFuture` function defined in future.h
/// converts any `Sender<Status, T>` into a `Future<T>`:
///
/// Heavily inspired by:
///
/// P1055: A Modest Executor Proposal
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1055r0.pdf
///
/// P1194: The Compromise Executors Proposal: A lazy simplification of P0443
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1194r0.html
///
/// The experimental implementation:
/// https://github.com/facebook/folly/tree/master/folly/experimental/pushmi

#include <atomic>
#include <iterator>
#include <tuple>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/utility/utility.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {

/// `NullReceiver` is a trivial implementation of the `Receiver` and
/// `FlowReceiver` concepts for which all customization points are no-ops.
class NullReceiver {
 public:
  template <typename CancelReceiver>
  friend void set_starting(NullReceiver&, CancelReceiver) {}

  template <typename... V>
  friend void set_value(NullReceiver&, V...) {}

  friend void set_done(NullReceiver&) {}

  template <typename E>
  friend void set_error(NullReceiver&, E e) {}

  friend void set_cancel(NullReceiver&) {}

  friend void set_stopping(NullReceiver&) {}
};

/// `NullSender` is a trivial implementation of the `Sender` and `FlowSender`
/// concepts for which `submit` is a no-op.
class NullSender {
  template <typename R>
  friend void submit(NullSender&, R) {}
};

/// Type-erased container for a move-only nullary function used by FlowReceiver
/// implementations to request cancellation.
using AnyCancelReceiver = internal::Poly<0, /*Copyable=*/false, void()>;

namespace internal_sender {

/// Used to implement `AnyReceiver` defined below.
template <typename E, typename... V>
using ReceiverPoly = internal::Poly<sizeof(void*) * 2, /*Copyable=*/false,
                                    void(internal_execution::set_value_t, V...),
                                    void(internal_execution::set_error_t, E),
                                    void(internal_execution::set_cancel_t)>;

/// Used to implement `AnyFlowReceiver` defined below.
template <typename E, typename... V>
using FlowReceiverPoly =
    internal::Poly<sizeof(void*) * 2, /*Copyable=*/false,
                   void(internal_execution::set_starting_t,
                        AnyCancelReceiver up),
                   void(internal_execution::set_value_t, V...),
                   void(internal_execution::set_done_t),
                   void(internal_execution::set_error_t, E),
                   void(internal_execution::set_stopping_t)>;
}  // namespace internal_sender

/// Type-erased container for any type that models the `Receiver<E, V...>`
/// concept.
template <typename E, typename... V>
class AnyReceiver : public internal_sender::ReceiverPoly<E, V...> {
  using Base = internal_sender::ReceiverPoly<E, V...>;

 public:
  /// Supports copy/move construction from any `Receiver<E, V...>` type.  Also
  /// supports in-place construction using `std::in_place_t<T>{}` as the first
  /// argument.
  using Base::Base;

  /// Constructs a null receiver.
  AnyReceiver() : Base(NullReceiver{}) {}

  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_value(V... v) {
    (*this)(internal_execution::set_value_t{}, std::forward<V>(v)...);
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_error(E e) {
    (*this)(internal_execution::set_error_t{}, std::forward<E>(e));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_cancel() {
    (*this)(internal_execution::set_cancel_t{});
  }
};

/// Type-erased container for any type that models `FlowReceiver<E, V...>`
/// concept.
template <typename E, typename... V>
class AnyFlowReceiver : public internal_sender::FlowReceiverPoly<E, V...> {
  using Base = internal_sender::FlowReceiverPoly<E, V...>;

 public:
  /// Supports copy/move construction from any `FlowReceiver<E, V...>` type.
  /// Also supports in-place construction using `std::in_place_t<T>{}` as the
  /// first argument.
  using Base::Base;

  /// Constructs a null receiver.
  AnyFlowReceiver() : Base(NullReceiver{}) {}

  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_starting(AnyCancelReceiver cancel) {
    (*this)(internal_execution::set_starting_t{}, std::move(cancel));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_value(V... v) {
    (*this)(internal_execution::set_value_t{}, std::forward<V>(v)...);
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_done() {
    (*this)(internal_execution::set_done_t{});
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_error(E e) {
    (*this)(internal_execution::set_error_t{}, std::forward<E>(e));
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE void set_stopping() {
    (*this)(internal_execution::set_stopping_t{});
  }
};

namespace internal_sender {

/// Used to implement `AnySender` defined below.
template <typename E, typename... V>
using SenderPoly =
    internal::Poly<(sizeof(V) + ... + 0), /*Copyable=*/false,
                   void(internal_execution::submit_t, AnyReceiver<E, V...>)>;

/// Used to implement `AnyFlowSender` defined below.
template <typename E, typename... V>
using FlowSenderPoly = internal::Poly<(sizeof(V) + ... + 0), /*Copyable=*/false,
                                      void(internal_execution::submit_t,
                                           AnyFlowReceiver<E, V...>)>;

}  // namespace internal_sender

/// Type-erased container for any type that models the `Sender<E, V...>`
/// concept.
template <typename E, typename... V>
class AnySender : public internal_sender::SenderPoly<E, V...> {
  using Base = internal_sender::SenderPoly<E, V...>;

 public:
  /// Supports copy/move construction from any `Sender<E, V...>` type.  Also
  /// supports in-place construction using `std::in_place_t<T>{}` as the first
  /// argument.
  using Base::Base;

  /// Constructs a null sender.
  AnySender() : Base(NullSender{}) {}

  ABSL_ATTRIBUTE_ALWAYS_INLINE void submit(AnyReceiver<E, V...> receiver) {
    (*this)(internal_execution::submit_t{}, std::move(receiver));
  }
};

/// Type-erased container for any type that models the `FlowSender<E, V...>`
/// concept.
template <typename E, typename... V>
class AnyFlowSender : public internal_sender::FlowSenderPoly<E, V...> {
  using Base = internal_sender::FlowSenderPoly<E, V...>;

 public:
  /// Supports copy/move construction from any `FlowSender<E, V...>` type.  Also
  /// supports in-place construction using `std::in_place_t<T>{}` as the first
  /// argument.
  using Base::Base;

  /// Constructs a null sender.
  AnyFlowSender() : Base(NullSender{}) {}

  ABSL_ATTRIBUTE_ALWAYS_INLINE void submit(AnyFlowReceiver<E, V...> receiver) {
    (*this)(internal_execution::submit_t{}, std::move(receiver));
  }
};

/// Sender that adapts an existing `sender` to invoke its `submit` function with
/// the specified `executor`.
template <typename Sender, typename Executor = tensorstore::Executor>
struct SenderWithExecutor {
  Executor executor;
  Sender sender;
  template <typename Receiver>
  void submit(Receiver receiver) {
    struct Callback {
      Sender sender;
      Receiver receiver;
      void operator()() { execution::submit(sender, std::move(receiver)); }
    };
    executor(Callback{std::move(sender), std::move(receiver)});
  }
};

/// Sender that immediately invokes `set_cancel`.
///
/// `CancelSender` is a model of `Sender<E, V...>` for any `E, V...`.
struct CancelSender {
  template <typename Receiver>
  friend void submit(CancelSender, Receiver receiver) {
    execution::set_cancel(receiver);
  }
};

/// Sender that immediately invokes `set_error` with the specified error.
///
/// `ErrorSender<E>` is a model of `Sender<E, V...>` for any `V...`.
template <typename E>
struct ErrorSender {
  E error;
  template <typename Receiver>
  friend void submit(ErrorSender& sender, Receiver receiver) {
    execution::set_error(receiver, std::move(sender.error));
  }
};
template <typename E>
ErrorSender(E error) -> ErrorSender<E>;

/// Sender that immediately invokes `set_value` with the specified values.
///
/// `ValueSender<V...>` is a model of `Sender<E, V...>` for any `E`.
template <typename... V>
struct ValueSender {
  ValueSender(V... v) : value(std::move(v)...) {}
  std::tuple<V...> value;
  template <typename Receiver>
  friend void submit(ValueSender& sender, Receiver receiver) {
    sender.SubmitHelper(std::move(receiver),
                        std::make_index_sequence<sizeof...(V)>{});
  }

 private:
  template <typename Receiver, std::size_t... Is>
  void SubmitHelper(Receiver receiver, std::index_sequence<Is...>) {
    execution::set_value(receiver, std::move(std::get<Is>(value))...);
  }
};
template <typename... V>
ValueSender(V... v) -> ValueSender<V...>;

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

#endif  // TENSORSTORE_UTIL_SENDER_H_
