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

#include <cstddef>
#include <limits>
#include <type_traits>

#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
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
/// \returns s[0] * ... * s[s.size()-1], or 1 if s.empty(), or
///     `std::numeric_limits<T>::max()` if integer overflow occurs.  However, if
///     any element is `0`, the return value is guaranteed to be `0`, even if
///     overflow occurred in computing an intermediate product.
template <typename T, std::ptrdiff_t Extent>
T ProductOfExtents(span<T, Extent> s) {
  using value_type = absl::remove_const_t<T>;
  value_type result = 1;
  for (const auto& x : s) {
    ABSL_ASSERT(x >= 0);
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
/// `span`-compatible with a `value_type` convertible without narrowing to
/// `Index` and a static `extent` compatible with `Rank`.
template <DimensionIndex Rank, typename Indices, typename = void>
struct IsCompatibleFullIndexVector : public std::false_type {};

template <DimensionIndex Rank, typename Indices>
struct IsCompatibleFullIndexVector<
    Rank, Indices, absl::void_t<internal::ConstSpanType<Indices>>>
    : public std::integral_constant<
          bool, AreStaticRanksCompatible(
                    Rank, internal::ConstSpanType<Indices>::extent) &&
                    internal::IsIndexPack<typename internal::ConstSpanType<
                        Indices>::value_type>::value> {};

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `value_type` convertible without narrowing to
/// `Index` and a static `extent` implicitly compatible with `Rank`.
template <DimensionIndex Rank, typename Indices, typename = void>
struct IsImplicitlyCompatibleFullIndexVector : public std::false_type {};

template <DimensionIndex Rank, typename Indices>
struct IsImplicitlyCompatibleFullIndexVector<
    Rank, Indices, absl::void_t<internal::ConstSpanType<Indices>>>
    : public std::integral_constant<
          bool, IsRankImplicitlyConvertible(
                    internal::ConstSpanType<Indices>::extent, Rank) &&
                    internal::IsIndexPack<typename internal::ConstSpanType<
                        Indices>::value_type>::value> {};

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `value_type` convertible without narrowing to
/// `Index` and a static `extent <= Rank`.
template <DimensionIndex Rank, typename Indices, typename = void>
struct IsCompatiblePartialIndexVector : public std::false_type {};

template <DimensionIndex Rank, typename Indices>
struct IsCompatiblePartialIndexVector<
    Rank, Indices, absl::void_t<internal::ConstSpanType<Indices>>>
    : public std::integral_constant<
          bool, IsStaticRankGreaterEqual(
                    Rank, internal::ConstSpanType<Indices>::extent) &&
                    internal::IsIndexPack<typename internal::ConstSpanType<
                        Indices>::value_type>::value> {};

/// Bool-valued metafunction that evaluates to `true` if every `IndexType` is
/// convertible without narrowing to `Index`, and `sizeof...(IndexType)` is
/// compatible with `Rank`.
template <DimensionIndex Rank, typename... IndexType>
struct IsCompatibleFullIndexPack
    : std::integral_constant<
          bool, AreStaticRanksCompatible(Rank, sizeof...(IndexType)) &&
                    internal::IsIndexPack<IndexType...>::value> {};

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `value_type` convertible without narrowing to
/// `Index`.
template <typename Indices, typename = void>
struct IsIndexConvertibleVector : public std::false_type {};

template <typename Indices>
struct IsIndexConvertibleVector<Indices,
                                absl::void_t<internal::ConstSpanType<Indices>>>
    : public internal::IsIndexPack<
          typename internal::ConstSpanType<Indices>::value_type> {};

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with a `value_type` of `Index`.
template <typename Indices, typename = Index>
struct IsIndexVector : public std::false_type {};

template <typename Indices>
struct IsIndexVector<Indices,
                     typename internal::ConstSpanType<Indices>::value_type>
    : public std::true_type {};

/// Bool-valued metafunction that evaluates to `true` if `Indices` is
/// `span`-compatible with an `element_type` of `Index`.
template <typename Indices, typename = Index>
struct IsMutableIndexVector : public std::false_type {};

template <typename Indices>
struct IsMutableIndexVector<Indices,
                            typename internal::SpanType<Indices>::element_type>
    : public std::true_type {};

namespace internal_extents {
/// Implementation detail for SpanStaticExtent.
template <typename... Xs>
struct SpanStaticExtentHelper {};

template <typename... Ts, std::ptrdiff_t Extent>
struct SpanStaticExtentHelper<span<Ts, Extent>...>
    : public std::integral_constant<std::ptrdiff_t, Extent> {};
}  // namespace internal_extents

/// `std::ptrdiff_t`-valued metafunction with a static constepxr `value` member
/// that is equal to the common static extent of `X0, Xs...` if `X0, Xs...` are
/// all `span`-compatible types with the same static extent.
///
/// If any of `X0, Xs...` are not `span`-compatible or do not have the same
/// static extent, there is no `value` member.
template <typename X0, typename... Xs>
using SpanStaticExtent =
    internal_extents::SpanStaticExtentHelper<internal::ConstSpanType<X0>,
                                             internal::ConstSpanType<Xs>...>;

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXTENTS_H_
