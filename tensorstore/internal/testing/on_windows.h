// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_TESTING_ON_WINDOWS_H_
#define TENSORSTORE_INTERNAL_TESTING_ON_WINDOWS_H_

#include <utility>

#include "absl/base/attributes.h"

namespace tensorstore {
namespace internal_testing {

/// Returns A on windows, B otherwise
#if defined(_WIN32)
template <typename A, typename B>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline A&& OnWindows(A&& a, B&&) {
  return std::forward<A>(a);
}
#else
template <typename A, typename B>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline B&& OnWindows(A&&, B&& b) {
  return std::forward<B>(b);
}
#endif

}  // namespace internal_testing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TESTING_ON_WINDOWS_H_
