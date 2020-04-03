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

#ifndef TENSORSTORE_UTIL_EXECUTION_H_
#define TENSORSTORE_UTIL_EXECUTION_H_

/// \file
/// Defines the customization points used by the Sender and Receiver concepts.
/// See `sender.h` for details.

#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {
namespace internal_execution {

/// Defines a function object type `NAME ## _t` that provides the interface for
/// a `tensorstore::execution` customization point named `NAME`.
///
/// In addition to handling the dispatch between the method vs free function,
/// this also defines a `PolyApply` overload that invokes the customization
/// point as well, which is used by the type-erased containers. The function
/// object class `NAME ## _t` itself serves as the first argument for tag
/// dispatching.
#define TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(NAME)         \
  TENSORSTORE_INTERNAL_DEFINE_HAS_METHOD(NAME)                                 \
  TENSORSTORE_INTERNAL_DEFINE_HAS_ADL_FUNCTION(NAME)                           \
  struct NAME##_t {                                                            \
    template <typename Self, typename... Arg>                                  \
    ABSL_ATTRIBUTE_ALWAYS_INLINE                                               \
        std::enable_if_t<HasMethod##NAME<void, Self&, Arg&&...>::value>        \
        operator()(Self&& self, Arg&&... arg) const {                          \
      self.NAME(std::forward<Arg>(arg)...);                                    \
    }                                                                          \
    template <typename Self, typename... Arg>                                  \
    ABSL_ATTRIBUTE_ALWAYS_INLINE                                               \
        std::enable_if_t<(!HasMethod##NAME<void, Self&, Arg&&...>::value &&    \
                          HasAdlFunction##NAME<void, Self&, Arg&&...>::value)> \
        operator()(Self&& self, Arg&&... arg) const {                          \
      NAME(self, std::forward<Arg>(arg)...);                                   \
    }                                                                          \
    template <typename Self, typename... Arg>                                  \
    friend ABSL_ATTRIBUTE_ALWAYS_INLINE decltype(std::declval<NAME##_t>()(     \
        std::declval<Self&>(), std::declval<Arg>()...))                        \
    PolyApply(Self& self, NAME##_t, Arg&&... arg) {                            \
      NAME##_t{}(self, std::forward<Arg>(arg)...);                             \
    }                                                                          \
  };                                                                           \
  /**/

TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(submit)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_starting)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_value)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_done)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_error)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_cancel)
TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT(set_stopping)

#undef TENSORSTORE_INTERNAL_DEFINE_EXECUTION_CUSTOMIZTION_POINT

}  // namespace internal_execution

namespace execution {
/// The public interface to these customization points is in the
/// `tensorstore::execution` namespace for consistency with the P1194 proposal
/// (linked above).  They can't be in the `tensorstore` namespace because they
/// would conflict with free functions of the same name, such as the friend
/// functions defined below for `NullReceiver`.
constexpr internal_execution::submit_t submit = {};
constexpr internal_execution::set_starting_t set_starting = {};
constexpr internal_execution::set_value_t set_value = {};
constexpr internal_execution::set_done_t set_done = {};
constexpr internal_execution::set_error_t set_error = {};
constexpr internal_execution::set_cancel_t set_cancel = {};
constexpr internal_execution::set_stopping_t set_stopping = {};
}  // namespace execution
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTION_H_
