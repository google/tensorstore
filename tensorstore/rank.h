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

#ifndef TENSORSTORE_RANK_H_
#define TENSORSTORE_RANK_H_

/// \file
/// The rank of multi-dimensional arrays can be specified either at compile time
/// or at run time.  To support this, functions and classes have `Rank` template
/// parameters of integer type `DimensionIndex` that specify "static rank"
/// values.  To specify a rank known at compile time, an integer >= 0 is
/// specified as the "static rank" value.  Otherwise, the special value of
/// `dynamic_rank = -1` is specified as the "static rank".

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

struct DynamicRank {
  /// Enables the use of `dynamic_rank` below as a constant argument for
  /// `DimensionIndex Rank` template parameters to indicate a dynamic rank (with
  /// inline buffer disabled, if applicable).
  constexpr operator DimensionIndex() const { return -1; }

  /// Enables the use of `dynamic_rank(n)` as an argument for
  /// `DimensionIndex Rank` template parameters to indicate a dynamic rank with
  /// an inline buffer of size `n`.
  constexpr DimensionIndex operator()(DimensionIndex inline_buffer_size) const {
    assert(inline_buffer_size >= 0);
    return -1 - inline_buffer_size;
  }
};

/// Special rank value indicating a rank specified at run time.
///
/// The value `dynamic_rank` is implicitly convertible to `DimensionIndex(-1)`,
/// which at compile-time represents a rank to be specified at run-time, and at
/// run-time represents an unknown rank.
///
/// The syntax `dynamic_rank(n)`, for `n >= 0`, is used to indicate a dynamic
/// rank with an inline buffer that accommodates ranks up to `n`.  Note that
/// `dynamic_rank` is equivalent to `dynamic_rank(0)`.
constexpr inline DynamicRank dynamic_rank = {};

constexpr inline DimensionIndex InlineRankLimit(DimensionIndex rank_spec) {
  return (rank_spec <= -1) ? -1 - rank_spec : 0;
}

constexpr inline DimensionIndex NormalizeRankSpec(DimensionIndex rank_spec) {
  return (rank_spec <= -1) ? -1 : rank_spec;
}

/// Returns `true` if `rank` is a valid static rank value.
constexpr inline bool IsValidStaticRank(DimensionIndex static_rank) {
  return static_rank >= 0 || static_rank == dynamic_rank;
}

/// Returns `true` if, and only if, a conversion from `source_rank` to
/// `dest_rank` is always valid.
constexpr inline bool IsRankImplicitlyConvertible(DimensionIndex source_rank,
                                                  DimensionIndex dest_rank) {
  return source_rank == dest_rank || dest_rank == dynamic_rank;
}

/// Returns `true`, if, and only if, a conversion from `source_rank` to
/// `dest_rank` is potentially valid (i.e. not known at compile-time to be
/// invalid).
constexpr inline bool IsRankExplicitlyConvertible(DimensionIndex source_rank,
                                                  DimensionIndex dest_rank) {
  return source_rank == dest_rank || source_rank == dynamic_rank ||
         dest_rank == dynamic_rank;
}

/// Template alias that evaluates to the type used for representing a static
/// rank.
///
/// This results in a substitution failure if `Rank < 0` (in particular, if
/// `Rank == dynamic_rank`).
///
/// If this is used to specify a default value for a function parameter, e.g.:
/// `StaticOrDynamicRank<Rank> rank = StaticRank<Rank>()`, the effect is that
/// the parameter is required if `Rank == dynamic_rank`, but optional otherwise.
template <DimensionIndex Rank>
using StaticRank =
    std::enable_if_t<Rank >= 0, std::integral_constant<DimensionIndex, Rank>>;

/// Template alias that evaluates to the type used for representing a static or
/// dynamic rank.
///
/// If `Rank == dynamic_rank`, this is `DimensionIndex`.  Otherwise, it is
/// `std::integral_constant<DimensionIndex, Rank>`.
template <DimensionIndex Rank>
using StaticOrDynamicRank =
    std::conditional_t<(Rank <= dynamic_rank), DimensionIndex,
                       std::integral_constant<DimensionIndex, Rank>>;

/// DimensionIndex-valued metafunction that evaluates to the static rank
/// corresponding to the specified RankType.  This metafunction is the inverse
/// of `StaticOrDynamicRank`.
///
/// If `RankType` is `DimensionIndex`, the result is
/// `std::integral_constant<DimensionIndex, dynamic_rank>`.  Otherwise, the
/// result is `RankType`.
///
/// \tparam RankType Either `DimensionIndex` or
///     `std::integral_constant<DimensionIndex, Rank>`.
template <typename RankType>
struct StaticRankFromRankType : public RankType {};

/// Specialization of `StaticRankFromRankType` for the case of a dynamic rank.
template <>
struct StaticRankFromRankType<DimensionIndex>
    : public std::integral_constant<DimensionIndex, dynamic_rank> {};

/// Returns the sum of two static rank values.
///
/// \pre `IsValidStaticRank(a) && IsValidStaticRank(b)`
/// \returns `dynamic_rank` if `a` or `b` equals `dynamic_rank`, else `a + b`.
constexpr DimensionIndex AddStaticRanks(DimensionIndex a, DimensionIndex b) {
  assert(IsValidStaticRank(a) && IsValidStaticRank(b));
  return a == dynamic_rank || b == dynamic_rank ? dynamic_rank : a + b;
}

/// Base case for zero arguments.
constexpr DimensionIndex AddStaticRanks() { return 0; }

/// Base case for a single argument.
constexpr DimensionIndex AddStaticRanks(DimensionIndex a) { return a; }

/// Returns the sum of a variable number of static rank values.
template <typename... T>
constexpr DimensionIndex AddStaticRanks(DimensionIndex a, DimensionIndex b,
                                        T... x) {
  return AddStaticRanks(a, AddStaticRanks(b, x...));
}

constexpr bool IsStaticRankLess(DimensionIndex a, DimensionIndex b) {
  return a == dynamic_rank || b == dynamic_rank || a < b;
}

constexpr bool IsStaticRankLessEqual(DimensionIndex a, DimensionIndex b) {
  return a == dynamic_rank || b == dynamic_rank || a <= b;
}

constexpr bool IsStaticRankGreater(DimensionIndex a, DimensionIndex b) {
  return a == dynamic_rank || b == dynamic_rank || a > b;
}

constexpr bool IsStaticRankGreaterEqual(DimensionIndex a, DimensionIndex b) {
  return a == dynamic_rank || b == dynamic_rank || a >= b;
}

/// Returns the sum of two static rank values.
///
/// \pre `IsValidStaticRank(a) && IsValidStaticRank(b) && (a == dynamic_rank ||
///     a >= b)`
/// \returns `dynamic_rank` if `a` or `b` equals `dynamic_rank`, else `a - b`.
constexpr DimensionIndex SubtractStaticRanks(DimensionIndex a,
                                             DimensionIndex b) {
  assert(IsValidStaticRank(a) && IsValidStaticRank(b) &&
         (a == dynamic_rank || a >= b));
  return (a == dynamic_rank || b == dynamic_rank ? dynamic_rank : a - b);
}

/// Returns the minimum statically-known rank value.
///
/// \pre `IsValidStaticRank(a) && IsValidStaticRank(b)`
/// \returns `a` if `b == dynamic_rank`, `b` if `a == dynamic_rank`, or `min(a,
///     b)` otherwise.
constexpr DimensionIndex MinStaticRank(DimensionIndex a, DimensionIndex b) {
  assert(IsValidStaticRank(a) && IsValidStaticRank(b));
  return (a == dynamic_rank || b == dynamic_rank ? (a < b ? b : a)
                                                 : (a < b ? a : b));
}

/// Base case for a single argument.
constexpr DimensionIndex MinStaticRank(DimensionIndex a) { return a; }

/// Base case for zero arguments.
constexpr DimensionIndex MinStaticRank() { return dynamic_rank; }

/// Returns the sum of a variable number of static rank values.
template <typename... T>
constexpr DimensionIndex MinStaticRank(DimensionIndex a, DimensionIndex b,
                                       T... x) {
  return MinStaticRank(a, MinStaticRank(b, x...));
}

/// Returns the maximum statically-known rank value.
///
/// \pre `IsValidStaticRank(a) && IsValidStaticRank(b)`
/// \returns `max(a,b)`
constexpr DimensionIndex MaxStaticRank(DimensionIndex a, DimensionIndex b) {
  assert(IsValidStaticRank(a) && IsValidStaticRank(b));
  return (a < b ? b : a);
}

/// Base case for zero arguments.
constexpr DimensionIndex MaxStaticRank() { return dynamic_rank; }

/// Base case for a single argument.
constexpr DimensionIndex MaxStaticRank(DimensionIndex a) {
  assert(IsValidStaticRank(a));
  return a;
}

/// Returns the max of a variable number of static rank values.
template <typename... T>
constexpr DimensionIndex MaxStaticRank(DimensionIndex a, DimensionIndex b,
                                       T... x) {
  return MaxStaticRank(a, MaxStaticRank(b, x...));
}

template <typename... T>
constexpr bool AreStaticRanksCompatible(T... rank) {
  return MinStaticRank(rank...) == MaxStaticRank(rank...);
}

/// If `Rank == dynamic_rank`, returns `dynamic_rank`.  Otherwise, returns
/// `StaticRank<Rank>{}`.
template <DimensionIndex Rank>
inline constexpr StaticOrDynamicRank<Rank> GetDefaultRank() {
  return {};
}

template <>
inline constexpr StaticOrDynamicRank<dynamic_rank>
GetDefaultRank<dynamic_rank>() {
  return dynamic_rank;
}

/// Evaluates to a type similar to `SourceRef` but with a static rank of
/// `TargetRank`.
///
/// The actual type is determined by the `RebindRank` template alias defined by
/// the `StaticCastTraits` specialization for `SourceRef`, which must be of the
/// following form:
///
///     template <DimensionIndex TargetRank>
///     using RebindRank = ...;
///
/// Supported types include `StaticOrDynamicRank<Rank>` (i.e. `DimensionIndex`
/// and `std::integral_constant<std::ptrdiff_t, Rank>`), `Array`,
/// `StridedLayout`, `Box`, `TransformedArray`, `Spec`, `TensorStore`.
///
/// \tparam SourceRef Optionally `const`- and/or reference-qualified source
///     type.  Any qualifiers are ignored.
/// \tparam TargetRank Target rank value.
template <typename SourceRef, DimensionIndex TargetRank>
using RebindRank =
    typename CastTraitsType<SourceRef>::template RebindRank<TargetRank>;

/// Casts `source` to have a static rank of `TargetRank`.
///
/// This simply uses `StaticCast` to convert `source` to the type obtained from
/// `RebindRank`.
///
/// The source type must be supported by `RebindRank`, and both the source and
/// target types must be supported by `StaticCast`.
///
/// The semantics of the `Checking` parameter are the same as for `StaticCast`.
///
/// Example:
///
///     StaticRank<3> r_static = StaticRankCast<3,
///     unchecked>(DimensionIndex(3)); Box<> box = ...; Result<Box<3>>
///     checked_result = StaticRankCast<3>(box); Box<3> unchecked_result =
///     StaticRankCast<3, unchecked>(box);
///
/// \tparam TargetRank Target rank value.
/// \tparam Checking Specifies whether the cast is checked or unchecked.
/// \param source Source value.
template <DimensionIndex TargetRank,
          CastChecking Checking = CastChecking::checked, typename SourceRef>
SupportedCastResultType<RebindRank<SourceRef, TargetRank>, SourceRef, Checking>
StaticRankCast(SourceRef&& source) {
  return StaticCast<RebindRank<SourceRef, TargetRank>, Checking>(
      std::forward<SourceRef>(source));
}

/// Specialization of `StaticCastTraits` for `DimensionIndex` (which is assumed
/// to represent a rank value).
template <>
struct StaticCastTraits<DimensionIndex>
    : public DefaultStaticCastTraits<DimensionIndex> {
  static constexpr DimensionIndex Construct(DimensionIndex rank) {
    return rank;
  }
  template <DimensionIndex Rank>
  static constexpr DimensionIndex Construct(
      std::integral_constant<DimensionIndex, Rank> rank) {
    return rank;
  }
  template <typename SourceRef>
  static constexpr bool IsCompatible(SourceRef&& source) {
    return true;
  }
  static std::string Describe(DimensionIndex value);

  // COV_NF_START
  static std::string Describe() { return Describe(dynamic_rank); }
  // COV_NF_END

  template <DimensionIndex TargetRank>
  using RebindRank = StaticOrDynamicRank<TargetRank>;
};

namespace internal_rank {
std::string DescribeStaticRank(DimensionIndex rank);
}  // namespace internal_rank

/// Specialization of `StaticCastTraits` for
/// `std::integral_constant<DimensionIndex, Rank>` (which is assumed to
/// represent a rank value).
template <DimensionIndex Rank>
struct StaticCastTraits<std::integral_constant<DimensionIndex, Rank>> {
  static constexpr StaticRank<Rank> Construct(StaticRank<Rank>) { return {}; }
  static constexpr StaticRank<Rank> Construct(DimensionIndex) { return {}; }
  static constexpr bool IsCompatible(DimensionIndex source) {
    return IsRankExplicitlyConvertible(source, Rank);
  }
  static std::string Describe() {
    return StaticCastTraits<DimensionIndex>::Describe(Rank);
  }
  // COV_NF_START
  static std::string Describe(StaticRank<Rank>) { return Describe(); }
  // COV_NF_END

  template <DimensionIndex TargetRank>
  using RebindRank = StaticOrDynamicRank<TargetRank>;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_RANK_H_
