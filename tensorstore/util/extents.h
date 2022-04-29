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

#ifndef TENSORSTORE_UTIL_EXTENTS_H_
#define TENSORSTORE_UTIL_EXTENTS_H_

#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "absl/base/optimization.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Computes the product of the elements of the given span.
///
/// \param s A contiguous sequence of numbers to multiply.
/// \dchecks All elements of `s` are non-negative.
/// \returns ``s[0] * ... * s[s.size()-1]``, or `1` if `s.empty()`, or
///     `std::numeric_limits<T>::max()` if integer overflow occurs.  However, if
///     any element is `0`, the return value is guaranteed to be `0`, even if
///     overflow occurred in computing an intermediate product.
///
/// \relates Box
template <typename T, std::ptrdiff_t Extent>
T ProductOfExtents(span<T, Extent> s) {
  using value_type = std::remove_const_t<T>;
  value_type result = 1;
  for (const auto& x : s) {
    assert(x >= 0);
    if (ABSL_PREDICT_FALSE(internal::MulOverflow(result, x, &result))) {
      // Overflow occurred.  We set the current product to
      // `std::numeric_limits<value_type>::max()`, but we don't return
      // immediately, because a subsequent extent of 0 can change the result.
      result = std::numeric_limits<value_type>::max();
    }
  }
  return result;
}

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::value_type` convertible without narrowing to
/// `Index` and a static `span::extent` compatible with `Rank`.
///
/// \ingroup index vectors
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsCompatibleFullIndexVector = false;

template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsCompatibleFullIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::EqualOrUnspecified(
        Rank, internal::ConstSpanType<Indices>::extent) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::value_type` convertible without narrowing to
/// `Index` and a static `span::extent` implicitly compatible with `Rank`.
///
/// \ingroup index vectors
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsImplicitlyCompatibleFullIndexVector = false;

template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsImplicitlyCompatibleFullIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::Implies(internal::ConstSpanType<Indices>::extent, Rank) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::value_type` convertible without narrowing to
/// `Index` and a static `span::extent <= Rank`.
///
/// \ingroup index vectors
template <DimensionIndex Rank, typename Indices, typename = void>
constexpr inline bool IsCompatiblePartialIndexVector = false;

template <DimensionIndex Rank, typename Indices>
constexpr inline bool IsCompatiblePartialIndexVector<
    Rank, Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    RankConstraint::GreaterEqualOrUnspecified(
        Rank, internal::ConstSpanType<Indices>::extent) &&
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;

/// Bool-valued metafunction that evaluates to `true` if every `IndexType` is
/// convertible without narrowing to `Index`, and `sizeof...(IndexType)` is
/// compatible with `Rank`.
///
/// \ingroup index vectors
template <DimensionIndex Rank, typename... IndexType>
constexpr inline bool IsCompatibleFullIndexPack =
    RankConstraint::EqualOrUnspecified(Rank, sizeof...(IndexType)) &&
    internal::IsIndexPack<IndexType...>;

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::value_type` convertible without narrowing to
/// `Index`.
//
/// \ingroup index vectors
template <typename Indices, typename = void>
constexpr inline bool IsIndexConvertibleVector = false;

template <typename Indices>
constexpr inline bool IsIndexConvertibleVector<
    Indices, std::void_t<internal::ConstSpanType<Indices>>> =
    internal::IsIndexPack<
        typename internal::ConstSpanType<Indices>::value_type>;

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::value_type` of `Index`.
///
/// \ingroup index vectors
template <typename Indices, typename = Index>
constexpr inline bool IsIndexVector = false;

template <typename Indices>
constexpr inline bool IsIndexVector<
    Indices, typename internal::ConstSpanType<Indices>::value_type> = true;

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `span::element_type` of `Index`.
///
/// \ingroup index vectors
template <typename Indices, typename = Index>
constexpr inline bool IsMutableIndexVector = false;

template <typename Indices>
constexpr inline bool IsMutableIndexVector<
    Indices, typename internal::SpanType<Indices>::element_type> = true;

namespace internal_extents {
/// Implementation detail for SpanStaticExtent.
template <typename... Xs>
struct SpanStaticExtentHelper {};

template <typename... Ts, std::ptrdiff_t Extent>
struct SpanStaticExtentHelper<span<Ts, Extent>...>
    : public std::integral_constant<std::ptrdiff_t, Extent> {};
}  // namespace internal_extents

/// `std::ptrdiff_t`-valued metafunction with a
/// ``static constepxr ptrdiff_t value`` member that is equal to the common
/// static extent of ``X0, Xs...`` if ``X0, Xs...`` are all
/// `span`-compatible types with the same static extent.
///
/// If any of ``X0, Xs...`` are not `span`-compatible or do not have the same
/// static extent, there is no ``value`` member.
///
/// \ingroup index vectors
template <typename X0, typename... Xs>
using SpanStaticExtent =
    internal_extents::SpanStaticExtentHelper<internal::ConstSpanType<X0>,
                                             internal::ConstSpanType<Xs>...>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXTENTS_H_
