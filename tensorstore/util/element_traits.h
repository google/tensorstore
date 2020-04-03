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

#ifndef TENSORSTORE_UTIL_ELEMENT_TRAITS_H_
#define TENSORSTORE_UTIL_ELEMENT_TRAITS_H_

#include <type_traits>

namespace tensorstore {

// Metafunction that evaluates to whether an array of Source-type elements is
// implicitly convertible to an array of Dest-type elements.
//
// Implicit conversions are (where T indicates a non-const type):
//
//   T          -> const T
//   T          ->       void
//   T          -> const void
//   const T    -> const void
//   void       -> const void
template <class Source, class Dest>
struct IsElementTypeImplicitlyConvertible
    : public std::integral_constant<
          bool, (std::is_const<Source>::value <= std::is_const<Dest>::value) &&
                    (std::is_same<const Source, const Dest>::value ||
                     std::is_void<Source>::value < std::is_void<Dest>::value)> {
};

// Metafunction that evaluates to whether an array of Source-type elements is
// explicitly BUT NOT implicitly convertible to an array of Dest-type elements.
//
// Explicit conversions are (where T indicates a non-const type):
//
//   const void -> const T
//   void       ->       T
//   void       -> const T
//
// Unlike the implicit conversions, these conversions are not statically known
// to be valid, and should be checked at run-time.
template <class Source, class Dest>
struct IsElementTypeOnlyExplicitlyConvertible
    : public std::integral_constant<bool, (std::is_void<Source>::value >
                                           std::is_void<Dest>::value) &&
                                              (std::is_const<Source>::value <=
                                               std::is_const<Dest>::value)> {};

/// Metafunction that evaluates to whether an array of Source-type elements is
/// implicitly or explicitly convertible to an array of Dest-type elements.
///
///   T -> T
///   T -> const T
///   T -> void
///   T -> const void
///   void -> T
///   void -> const T
///   const void -> const T
template <typename Source, typename Dest>
struct IsElementTypeExplicitlyConvertible
    : public std::integral_constant<
          bool, (std::is_const<Source>::value <= std::is_const<Dest>::value) &&
                    (std::is_void<Source>::value || std::is_void<Dest>::value ||
                     std::is_same<const Source, const Dest>::value)> {};

/// Metafunction that evaluates to whether A and B could refer to the same type,
/// ignoring const.
///
/// (T, T) -> true
/// (const T, T) -> true
/// (T, const T) -> true
/// (const T, const T) -> true
/// (const T, void) -> true
/// (const T, const void) -> true
/// (T, void) -> true
/// (T, const void) -> true
/// (void, const T) -> true
/// (const void, const T) -> true
/// (void, T) -> true
/// (const void, T) -> true
/// (T, U) -> false
/// (const T, U) -> false
/// (T, const U) -> false
/// (const T, const U) -> false
template <typename A, typename B>
struct AreElementTypesCompatible
    : public std::integral_constant<
          bool, (std::is_void<A>::value || std::is_void<B>::value ||
                 std::is_same<const A, const B>::value)> {};

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ELEMENT_TRAITS_H_
