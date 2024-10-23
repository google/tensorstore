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

#include <type_traits>
#include <utility>

#include "absl/meta/type_traits.h"
#include "absl/status/status.h"

namespace tensorstore {

/// Checks if every `Option` is compatible with the specified `Options` type.
///
/// \ingroup utilities
template <typename Options, typename... Option>
// NONITPICK: Options::IsOption<std::remove_cvref_t<Option>>
constexpr inline bool IsCompatibleOptionSequence =
    ((!std::is_same_v<Options, absl::remove_cvref_t<Option>> &&
      Options::template IsOption<absl::remove_cvref_t<Option>>) &&
     ...);

namespace internal {

/// Collects a parameter pack of options into an options type.
///
/// An Options type is a struct which accepts types via a`Set` method;
/// Each type has a corresponding `constexpr bool IsOption<T>` which evaluates
/// to true for each settable type.
///
/// \ingroup utilities
/// Example usage:
///
///    OpenOptions options;
///    TENSORSTORE_RETURN_IF_ERROR
///        SetAll(options, std::forward<Args>(args)...));
///
template <typename Options, typename... Args>
absl::Status SetAll(Options& options, Args&&... args) {
  absl::Status status;
  ((status.Update(options.Set(std::forward<Args>(args)))), ...);
  return status;
}
template <typename Options>
absl::Status SetAll(Options& options) {
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_OPTION_H_
