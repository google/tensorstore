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

#ifndef TENSORSTORE_INTERNAL_ATTRIBUTES_H_
#define TENSORSTORE_INTERNAL_ATTRIBUTES_H_

#include "absl/base/attributes.h"

#ifndef TENSORSTORE_CPP_DOC_GENERATION
#define TENSORSTORE_ATTRIBUTE_LIFETIME_BOUND ABSL_ATTRIBUTE_LIFETIME_BOUND
#else
// Expand to nothing to avoid cluttering the documentation and because the
// Sphinx C++ parser chokes on it.
#define TENSORSTORE_ATTRIBUTE_LIFETIME_BOUND
#endif

// When defining a conditionally-explicit constructor, this may be used on the
// explicit overload to wrap the portion of the `enable_if` conditions that
// differ from the implicit overload.
//
// For example:
//
//     template <typename T>
//     struct Foo {
//
//       /// Constructs from a convertible value.
//       template <typename U, typename = std::enable_if_t<
//           (std::is_constructible_v<T, U> &&
//            ExplicitRequires(!std::is_convertible_v<U, T>)>>
//       explicit Foo(U &&value);
//
//       // Undocumented
//       template <typename U, typename = std::enable_if_t<
//           std::is_convertible_v<U, T>>>
//       Foo(U &&value);
//     };
//
// When the API documentation is generated, this will be treated as equivalent
// to the C++20 syntax:
//
//     template <typename T>
//     struct Foo {
//
//       /// Constructs from a convertible value.
//       template <typename U>
//           requires(std::is_constructible_v<T, U>)
//       explicit(!std::is_convertible_v<U, T>)
//       Foo(U &&value);
//     };
namespace tensorstore {
constexpr inline bool ExplicitRequires(bool value) { return value; }
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ATTRIBUTES_H_
