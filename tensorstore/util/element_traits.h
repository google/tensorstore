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

/// Metafunction that evaluates to `true` if an array of `Source`-type elements
/// is implicitly convertible to an array of `Dest`-type elements.
///
/// Implicit conversions are (where ``T`` indicates a non-const type):
///
/// =========== ==============
/// `Source`    `Dest`
/// =========== ==============
/// ``T``       ``const T``
/// ``T``       ``void``
/// ``T``       ``const void``
/// ``const T`` ``const void``
/// ``void``    ``const void``
/// =========== ==============
///
/// \relates ElementPointer
/// \membergroup Metafunctions
template <typename Source, typename Dest>
constexpr inline bool IsElementTypeImplicitlyConvertible =
    (std::is_const_v<Source> <= std::is_const_v<Dest>)&&  //
    (std::is_same_v<const Source, const Dest> ||
     std::is_void_v<Source> < std::is_void_v<Dest>);

/// Metafunction that evaluates `true` if an array of `Source`-type elements is
/// explicitly BUT NOT implicitly convertible to an array of `Dest`-type
/// elements.
///
/// Explicit conversions are (where ``T`` indicates a non-const type):
///
/// =============== ==============
/// `Source`        `Dest`
/// =============== ==============
/// ``const void``  ``const T``
/// ``void``        ``T``
/// ``void``        ``const T``
/// =============== ==============
///
/// Unlike the implicit conversions, these conversions are not statically known
/// to be valid, and should be checked at run-time.
///
/// \relates ElementPointer
/// \membergroup Metafunctions
template <class Source, class Dest>
constexpr inline bool IsElementTypeOnlyExplicitlyConvertible =
    (std::is_void_v<Source> > std::is_void_v<Dest>)&&  //
    (std::is_const_v<Source> <= std::is_const_v<Dest>);

/// Metafunction that evaluates to whether an array of Source-type elements is
/// implicitly or explicitly convertible to an array of Dest-type elements.
///
/// ============== ==============
/// `Source`       `Dest`
/// ============== ==============
/// ``T``          ``T``
/// ``T``          ``const T``
/// ``T``          ``void``
/// ``T``          ``const void``
/// ``void``       ``T``
/// ``void``       ``const T``
/// ``const void`` ``const T``
/// ============== ==============
///
/// \relates ElementPointer
/// \membergroup Metafunctions
template <typename Source, typename Dest>
constexpr inline bool IsElementTypeExplicitlyConvertible =
    (std::is_const_v<Source> <= std::is_const_v<Dest>)&&  //
    (std::is_void_v<Source> || std::is_void_v<Dest> ||
     std::is_same_v<const Source, const Dest>);

/// Metafunction that evaluates to whether `A` and `B` could refer to the same
/// type, ignoring const.
///
/// ============== ============== ============
/// `A`            `B`            Compatible
/// ============== ============== ============
/// ``T``          ``T``          ``true``
/// ``const T``    ``T``          ``true``
/// ``T``          ``const T``    ``true``
/// ``const T``    ``const T``    ``true``
/// ``const T``    ``void``       ``true``
/// ``const T``    ``const void`` ``true``
/// ``T``          ``void``       ``true``
/// ``T``          ``const void`` ``true``
/// ``void``       ``const T``    ``true``
/// ``const void`` ``const T``    ``true``
/// ``void``       ``T``          ``true``
/// ``const void`` ``T``          ``true``
/// ``T``          ``U``          ``false``
/// ``const T``    ``U``          ``false``
/// ``T``          ``const U``    ``false``
/// ``const T``    ``const U``    ``false``
/// ============== ============== ============
///
/// \relates ElementPointer
/// \membergroup Metafunctions
template <typename A, typename B>
constexpr inline bool AreElementTypesCompatible =
    (std::is_void_v<A> || std::is_void_v<B> ||
     std::is_same_v<const A, const B>);

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ELEMENT_TRAITS_H_
