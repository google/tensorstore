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

#ifndef TENSORSTORE_UTIL_EXECUTION_EXECUTION_H_
#define TENSORSTORE_UTIL_EXECUTION_EXECUTION_H_

/// \file
/// Defines the customization points used by the Sender and Receiver concepts.
/// See `sender.h` for details.

#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"

namespace tensorstore {
namespace internal_execution {

/// C++ detector idiom, simplified
template <template <class...> class Trait, class AlwaysVoid, class... Arg>
struct detector_impl : std::false_type {};

template <template <class...> class Trait, class... Arg>
struct detector_impl<Trait, std::void_t<Trait<Arg...>>, Arg...>
    : std::true_type {};

template <template <class...> class Trait, class... Args>
using detected_t = typename detector_impl<Trait, void, Args...>::type;

/// TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(name)
///
/// Defines the internals of a function object type (which must be named
/// `NAME ## _t`) which provides the interface for a `tensorstore::execution`
/// customization point named `NAME`.
///
/// In addition to handling the dispatch between the method vs free function,
/// this also defines a `PolyApply` overload that invokes the customization
/// point as well, which is used by the type-erased containers. The function
/// object class `NAME ## _t` itself serves as the first argument for tag
/// dispatching.
///
#define TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(NAME)               \
 private:                                                                    \
  /* utility types: return_NAME, return_adl_NAME, has_NAME, has_adl_NAME  */ \
  template <typename T, typename... Arg>                                     \
  using return_##NAME =                                                      \
      decltype(std::declval<T&&>().NAME(std::declval<Arg>()...));            \
  template <typename... Arg>                                                 \
  using return_adl_##NAME = decltype(NAME(std::declval<Arg>()...));          \
                                                                             \
  template <typename T, typename... Arg>                                     \
  using has_##NAME = detected_t<return_##NAME, T, Arg...>;                   \
  template <typename T, typename... Arg>                                     \
  using has_adl_##NAME = detected_t<return_adl_##NAME, T, Arg...>;           \
                                                                             \
 public:                                                                     \
  template <typename Self, typename... Arg>                                  \
  ABSL_ATTRIBUTE_ALWAYS_INLINE std::enable_if_t<                             \
      has_##NAME<Self&, Arg&&...>::value, return_##NAME<Self&, Arg&&...>>    \
  operator()(Self&& self, Arg&&... arg) const {                              \
    return self.NAME(std::forward<Arg>(arg)...);                             \
  }                                                                          \
  template <typename Self, typename... Arg>                                  \
  ABSL_ATTRIBUTE_ALWAYS_INLINE                                               \
      std::enable_if_t<(!has_##NAME<Self&, Arg&&...>::value &&               \
                        has_adl_##NAME<Self&, Arg&&...>::value),             \
                       return_adl_##NAME<Self&, Arg&&...>>                   \
      operator()(Self&& self, Arg&&... arg) const {                          \
    return NAME(self, std::forward<Arg>(arg)...);                            \
  }                                                                          \
  template <typename Self, typename... Arg>                                  \
  friend ABSL_ATTRIBUTE_ALWAYS_INLINE decltype(std::declval<NAME##_t>()(     \
      std::declval<Self&>(), std::declval<Arg>()...))                        \
  PolyApply(Self& self, NAME##_t, Arg&&... arg) {                            \
    return NAME##_t{}(self, std::forward<Arg>(arg)...);                      \
  }

struct submit_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(submit)
};
struct set_starting_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_starting)
};
struct set_value_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_value)
};
struct set_done_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_done)
};
struct set_error_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_error)
};
struct set_cancel_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_cancel)
};
struct set_stopping_t {
  TENSORSTORE_DEFINE_EXECUTION_CUSTOMIZATION_POINT(set_stopping)
};

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

#endif  // TENSORSTORE_UTIL_EXECUTION_EXECUTION_H_
