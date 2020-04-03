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

/// Basic metaprogramming facilities.

#ifndef TENSORSTORE_INTERNAL_META_H_
#define TENSORSTORE_INTERNAL_META_H_

namespace tensorstore {
namespace internal {

/// Returns the first argument, useful for decomposing a parameter pack.
template <typename T, typename... Ts>
constexpr T&& GetFirstArgument(T&& t, Ts&&... ts) {
  return static_cast<T&&>(t);
}

// Intentionally not marked constexpr to trigger error during constexpr
// evaluation.
inline int constexpr_assert_failed() noexcept { return 0; }

/// Works like `assert`, but only triggers during constexpr evaluation.
#define TENSORSTORE_CONSTEXPR_ASSERT(...) \
  (static_cast<void>(                     \
      (__VA_ARGS__) ? 0                   \
                    : tensorstore::internal::constexpr_assert_failed())) /**/

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_META_H_
