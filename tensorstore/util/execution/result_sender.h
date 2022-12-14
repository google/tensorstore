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

#ifndef TENSORSTORE_UTIL_EXECUTION_RESULT_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_RESULT_SENDER_H_

#include <functional>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_result {

/// Detector for whether Receiver is compatible with Result<T>.
template <typename Receiver, typename = void, typename = void, typename = void,
          typename = void>
struct IsResultReceiver : public std::false_type {};

template <typename Receiver, typename T>
struct IsResultReceiver<
    Receiver, T,
    decltype(execution::set_value(std::declval<Receiver&>(),
                                  std::declval<T>())),
    decltype(execution::set_error(std::declval<Receiver&>(),
                                  std::declval<absl::Status>())),
    decltype(execution::set_cancel(std::declval<Receiver&>()))>
    : public std::true_type {};

}  // namespace internal_result

/// The `set_value`, `set_cancel`, `set_error`, and `submit` functions defined
/// below make `Result<T>` model the `Receiver<absl::Status, T>`.
template <typename T, typename... V>
std::enable_if_t<((std::is_same_v<void, T> && sizeof...(V) == 0) ||
                  std::is_constructible_v<T, V&&...>)>
set_value(Result<T>& r, V&&... v) {
  r.emplace(std::forward<V>(v)...);
}
template <typename T, typename... V>
std::enable_if_t<((std::is_same_v<void, T> && sizeof...(V) == 0) ||
                  std::is_constructible_v<T, V&&...>)>
set_value(std::reference_wrapper<Result<T>> r, V&&... v) {
  set_value(r.get(), std::forward<V>(v)...);
}

// Implements the Receiver `set_error` operation.
//
// Overrides the existing value/error with `status`.
template <typename T>
void set_error(Result<T>& r, absl::Status status) {
  r = std::move(status);
}
template <typename T>
void set_error(std::reference_wrapper<Result<T>> r, absl::Status status) {
  set_error(r.get(), std::move(status));
}

// Implements the Receiver `set_cancel` operation.
//
// This overrides the existing value/error with `absl::CancelledError("")`.
template <typename T>
void set_cancel(Result<T>& r) {
  r = absl::CancelledError("");
}
template <typename T>
void set_cancel(std::reference_wrapper<Result<T>> r) {
  set_cancel(r.get());
}

/// A Sender for `Result<T>` modelling the `Sender<absl::Status, T>` concept.
///
/// If `has_value() == true`, calls `set_value` with an lvalue reference to
/// the contained value.
///
/// If in an error state with an error code of `absl::StatusCode::kCancelled`,
/// calls `set_cancel`.
///
/// Otherwise, calls `set_error` with `status()`.
template <typename T, typename Receiver>
std::enable_if_t<internal_result::IsResultReceiver<Receiver, T>::value>  //
submit(Result<T>& r, Receiver&& receiver) {
  if (r.has_value()) {
    execution::set_value(receiver, r.value());
  } else {
    auto status = r.status();
    if (status.code() == absl::StatusCode::kCancelled) {
      execution::set_cancel(receiver);
    } else {
      execution::set_error(receiver, std::move(status));
    }
  }
}

template <typename T, typename Receiver>
std::enable_if_t<internal_result::IsResultReceiver<Receiver, T>::value>  //
submit(std::reference_wrapper<Result<T>> r, Receiver&& receiver) {
  submit(r.get(), std::forward<Receiver>(receiver));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_RESULT_SENDER_H_
