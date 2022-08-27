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

#ifndef TENSORSTORE_UTIL_EXECUTION_ANY_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_ANY_SENDER_H_

#include <utility>

#include "absl/base/attributes.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender.h"

namespace tensorstore {
namespace internal_sender {

/// Used to implement `AnySender` defined below.
template <typename E, typename... V>
using SenderPoly =
    poly::Poly<(sizeof(V) + ... + 0), /*Copyable=*/false,
               void(internal_execution::submit_t, AnyReceiver<E, V...>)>;

/// Used to implement `AnyFlowSender` defined below.
template <typename E, typename... V>
using FlowSenderPoly =
    poly::Poly<(sizeof(V) + ... + 0), /*Copyable=*/false,
               void(internal_execution::submit_t, AnyFlowReceiver<E, V...>)>;

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


}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_SENDER_H_
