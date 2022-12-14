// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_UTIL_EXECUTION_FUTURE_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_FUTURE_SENDER_H_

#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_future {

/// Detector for whether Receiver is compatible with Future<T>.
template <typename Receiver, typename = void, typename = void, typename = void,
          typename = void>
struct IsFutureReceiver : public std::false_type {};

template <typename Receiver, typename T>
struct IsFutureReceiver<
    Receiver, T,
    decltype(execution::set_value(std::declval<Receiver&>(),
                                  std::declval<T>())),
    decltype(execution::set_error(std::declval<Receiver&>(),
                                  std::declval<absl::Status>())),
    decltype(execution::set_cancel(std::declval<Receiver&>()))>
    : public std::true_type {};

}  // namespace internal_future

/// A Receiver for `Future<T>` modelling the `Receiver<absl::Status, T>`
/// concept.
///
/// The `set_value`, `set_error`, and `set_cancel` functions defined below
/// make `Promise<T>` model the `Receiver<absl::Status, T>` concept.  Calling
/// any of these methods has no effect if the promise is already in a ready
/// state. This implies that calling any of these functions after they have
/// already been called on a given Promise has no effect.

/// Implements the Receiver `set_value` operation.
template <typename T, typename... V>
std::enable_if_t<(!std::is_const_v<T> &&
                  std::is_constructible_v<typename Promise<T>::result_type,
                                          std::in_place_t, V...>)>
set_value(const Promise<T>& promise, V&&... v) {
  promise.SetResult(std::in_place, std::forward<V>(v)...);
}

template <typename T, typename... V>
std::enable_if_t<(!std::is_const_v<T> &&
                  std::is_constructible_v<typename Promise<T>::result_type,
                                          std::in_place_t, V...>)>
set_value(std::reference_wrapper<const Promise<T>> promise, V&&... v) {
  set_value(promise.get(), std::forward<V>(v)...);
}

/// Implements the Receiver `set_error` operation.
template <typename T>
void set_error(const Promise<T>& promise, absl::Status error) {
  promise.SetResult(std::move(error));
}
template <typename T>
void set_error(std::reference_wrapper<const Promise<T>> promise,
               absl::Status error) {
  set_error(promise.get(), std::move(error));
}

/// Implements the Receiver `set_cancel` operation.
template <typename T>
void set_cancel(const Promise<T>& promise) {
  promise.SetResult(absl::CancelledError(""));
}
template <typename T>
void set_cancel(std::reference_wrapper<const Promise<T>> promise) {
  set_cancel(promise.get());
}

// A Sender for `Future<T>` modelling the `Sender<absl::Status, T>` concept.
//
// If `has_value() == true`, calls `set_value` with an lvalue reference to
// the contained value.
//
// If in an error state with an error code of `absl::StatusCode::kCancelled`,
// calls `set_cancel`.
//
// Otherwise, calls `set_error` with `status()`.
template <typename T, typename Receiver>
std::enable_if_t<internal_future::IsFutureReceiver<Receiver, T>::value>  //
submit(Future<T>& f, Receiver receiver) {
  f.Force();
  f.ExecuteWhenReady([r = std::move(receiver)](ReadyFuture<T> ready) mutable {
    auto& result = ready.result();
    if (result.has_value()) {
      execution::set_value(r, result.value());
    } else {
      auto status = ready.status();
      if (status.code() == absl::StatusCode::kCancelled) {
        execution::set_cancel(r);
      } else {
        execution::set_error(r, std::move(status));
      }
    }
  });
}

template <typename T, typename Receiver>
std::enable_if_t<internal_future::IsFutureReceiver<Receiver, T>::value>  //
submit(std::reference_wrapper<Future<T>> f, Receiver&& receiver) {
  submit(f.get(), std::forward<Receiver>(receiver));
}

// Converts an arbitrary `Sender<absl::Status, T>` into a `Future<T>`.
template <typename T, typename Sender>
Future<T> MakeSenderFuture(Sender sender) {
  auto pair = PromiseFuturePair<T>::Make();
  struct Callback {
    Sender sender;
    void operator()(Promise<T> promise) { execution::submit(sender, promise); }
  };
  pair.promise.ExecuteWhenForced(Callback{std::move(sender)});
  return pair.future;
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_FUTURE_SENDER_H_
