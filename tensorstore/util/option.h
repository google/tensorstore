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

#ifndef TENSORSTORE_UTIL_OPTION_H_
#define TENSORSTORE_UTIL_OPTION_H_

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {

/// Checks if every `Option` is compatible with the specified `Options` type.
///
/// \ingroup utilities
template <typename Options, typename... Option>
// NONITPICK: Options::IsOption<std::remove_cvref_t<Option>>
constexpr inline bool IsCompatibleOptionSequence =
    (Options::template IsOption<internal::remove_cvref_t<Option>> && ...);

}  // namespace tensorstore

/// Collects a parameter pack of options into an options type.
///
/// Intended to be used in a function scope.  Defines a local variable named
/// `OPTIONS_NAME` of type `OPTIONS_TYPE`, and attempts to set each of the
/// options specified by `OPTIONS_PACK`.  If setting an option fails with an
/// error status, it is returned.
///
/// Example usage:
///
///     template <typename... Option>
///     std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
///                      Result<Whatever>>
///     MyFunction(Option&&... option) {
///       TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(
///           OpenOptions, options, option);
///       // use `options`
///     }
#define TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(          \
    OPTIONS_TYPE, OPTIONS_NAME, OPTION_PACK)                    \
  OPTIONS_TYPE OPTIONS_NAME;                                    \
  if (absl::Status status;                                      \
      !((status = OPTIONS_NAME.Set(                             \
             std::forward<decltype(OPTION_PACK)>(OPTION_PACK))) \
            .ok() &&                                            \
        ...)) {                                                 \
    return status;                                              \
  }                                                             \
  /**/

#endif  // TENSORSTORE_UTIL_OPTION_H_
