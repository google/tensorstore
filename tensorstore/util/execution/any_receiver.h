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

#ifndef TENSORSTORE_UTIL_EXECUTION_ANY_RECEIVER_H_
#define TENSORSTORE_UTIL_EXECUTION_ANY_RECEIVER_H_

#include <utility>

#include "absl/base/attributes.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"

namespace tensorstore {

/// Type-erased container for a move-only nullary function used by FlowReceiver
/// implementations to request cancellation.
using AnyCancelReceiver = poly::Poly<0, /*Copyable=*/false, void()>;

namespace internal_sender {

/// Used to implement `AnyReceiver` defined below.
template <typename E, typename... V>
using ReceiverPoly = poly::Poly<sizeof(void*) * 2, /*Copyable=*/false,
                                void(internal_execution::set_value_t, V...),
                                void(internal_execution::set_error_t, E),
                                void(internal_execution::set_cancel_t)>;

/// Used to implement `AnyFlowReceiver` defined below.
template <typename E, typename... V>
using FlowReceiverPoly =
    poly::Poly<sizeof(void*) * 2, /*Copyable=*/false,
               void(internal_execution::set_starting_t, AnyCancelReceiver up),
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

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_ANY_RECEIVER_H_
