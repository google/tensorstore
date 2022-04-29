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

#if _MSC_FULL_VER >= 192829913
// https://devblogs.microsoft.com/cppblog/msvc-cpp20-and-the-std-cpp20-switch/#c20-no_unique_address
// On VS 2019 v16.9 and later, `msvc::no_unique_address` performs the
// optimization in /std:c++14 and /std:c++17 language modes as well.
#define TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif ABSL_HAVE_ATTRIBUTE(no_unique_address)
#define TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif

// TENSORSTORE_LIFETIME_BOUND
//
// A clang-specific c++ attribute that indicates that the lifetime of a
// function result is bound to one of the arguments. This is particularly
// useful when constructing reference-like types such as tensorstore::span.
//
// See https://wg21.link/p0936r0, https://reviews.llvm.org/D49922
#ifndef TENSORSTORE_LIFETIME_BOUND
#if defined(__cplusplus) && defined(__has_cpp_attribute) && \
    __has_cpp_attribute(clang::lifetimebound)
#define TENSORSTORE_LIFETIME_BOUND [[clang::lifetimebound]]
#else
#define TENSORSTORE_LIFETIME_BOUND
#endif
#endif  // !defined(TENSORSTORE_LIFETIME_BOUND)

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
