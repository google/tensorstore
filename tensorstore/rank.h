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
#include <initializer_list>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Maximum supported rank.
///
/// \relates DimensionIndex
constexpr DimensionIndex kMaxRank = 32;

/// Checks if `rank` is a valid rank value.
///
/// \relates DimensionIndex
constexpr inline bool IsValidRank(DimensionIndex rank) {
  return 0 <= rank && rank <= kMaxRank;
}

/// Type of the special `dynamic_rank` constant.
///
/// \relates dynamic_rank
struct DynamicRank {
  /// Enables the use of `dynamic_rank` below as a constant argument for
  /// ``DimensionIndex Rank`` template parameters to indicate a dynamic rank
  /// (with inline buffer disabled, if applicable).
  constexpr operator DimensionIndex() const { return -1; }

  /// Enables the use of ``dynamic_rank(n)`` as an argument for
  /// ``DimensionIndex Rank`` template parameters to indicate a dynamic rank
  /// with an inline buffer of size ``n``.
  constexpr DimensionIndex operator()(DimensionIndex inline_buffer_size) const {
    assert(inline_buffer_size >= 0);
    assert(inline_buffer_size <= kMaxRank);
    return -1 - inline_buffer_size;
  }
};

/// Special rank value indicating a rank specified at run time.
///
/// The value `dynamic_rank` is implicitly convertible to `DimensionIndex(-1)`,
/// which at compile-time represents a rank to be specified at run-time, and at
/// run-time represents an unknown rank.
///
/// The syntax ``dynamic_rank(n)``, for ``n >= 0``, is used to indicate a
/// dynamic rank with an inline buffer that accommodates ranks up to ``n``.
/// Note that `dynamic_rank` is equivalent to `dynamic_rank(0)`.
///
/// \relates DimensionIndex
constexpr inline DynamicRank dynamic_rank = {};

/// Specifies a fixed compile-time rank, or a run-time rank with an optional
/// inline storage limit.
///
/// - Non-negative integers `0`, `1`, ..., `kMaxRank` indicate a fixed rank
///   specified at compile time.
///
/// - A value of ``dynamic_rank(k)`` (equal to ``-k -1``), for
///   ``0 <= k <= kMaxRank``, indicates that the rank is determined at run
///   time, and bounds/layout information for rank values up to ``k`` will be
///   stored inline without additional heap allocation.
///
/// - The special value `dynamic_rank`, equivalent to `dynamic_rank(0)`,
///   indicates that the rank is determined at run time, and any non-zero rank
///   will require heap-allocated storage of bounds/layout information.
///
/// \ingroup Indexing
using InlineRank = DimensionIndex;

/// Represents an optional rank value and provides related operations.
///
/// \ingroup Indexing
struct RankConstraint {
  /// Constructs with an unspecified rank value (`dynamic_rank`).
  ///
  /// \id dynamic
  constexpr RankConstraint() = default;
  constexpr RankConstraint(DynamicRank) {}

  /// Constructs with the specified rank value.
  ///
  /// \id rank
  constexpr explicit RankConstraint(DimensionIndex rank) : rank(rank) {}

  /// Constructs from an `InlineRank` value, ignoring the inline buffer size.
  ///
  /// \membergroup Constructors
  static constexpr RankConstraint FromInlineRank(InlineRank value) {
    return RankConstraint(value < 0 ? dynamic_rank : value);
  }

  /// Indicates the rank, or equal to `dynamic_rank` if unknown.
  DimensionIndex rank = dynamic_rank;

  /// Returns `rank`.
  constexpr operator DimensionIndex() const { return rank; }

  /// Returns `true` if this is a valid rank constraint.
  constexpr bool valid() const {
    return rank == -1 || (rank >= 0 && rank <= kMaxRank);
  }

  /// Returns the intersection of the rank constraints.
  ///
  /// \pre `EqualOrUnspecified(a, b)` or `EqualOrUnspecified(constraints)`
  ///
  /// \membergroup Composition
  static constexpr RankConstraint And(DimensionIndex a, DimensionIndex b) {
    assert(EqualOrUnspecified(a, b));
    return RankConstraint(a == dynamic_rank ? b : a);
  }
  static constexpr RankConstraint And(
      std::initializer_list<DimensionIndex> constraints) {
    assert(EqualOrUnspecified(constraints));
    for (DimensionIndex x : constraints) {
      if (x == dynamic_rank) continue;
      return RankConstraint(x);
    }
    return dynamic_rank;
  }

  /// Adds the rank constraints.
  ///
  /// - If any constraint is equal to `dynamic_rank`, the result is
  ///   `dynamic_rank`.
  ///
  /// - Otherwise, the result is the sum of the two fixed ranks.
  ///
  /// \membergroup Composition
  static constexpr RankConstraint Add(DimensionIndex a, DimensionIndex b) {
    if (a == dynamic_rank || b == dynamic_rank) return dynamic_rank;
    return RankConstraint(a + b);
  }
  static constexpr RankConstraint Add(
      std::initializer_list<DimensionIndex> constraints) {
    DimensionIndex result = 0;
    for (auto x : constraints) {
      if (x == dynamic_rank) return dynamic_rank;
      result += x;
    }
    return RankConstraint(result);
  }

  /// Subtracts the rank constraints.
  ///
  /// - If `a` or `b` is equal to `dynamic_rank`, the result is `dynamic_rank`.
  ///
  /// - Otherwise, the result is equal to the difference of the two fixed ranks.
  ///
  /// \pre `GreaterEqualOrUnspecified(a, b)`
  ///
  /// \membergroup Composition
  static constexpr RankConstraint Subtract(DimensionIndex a, DimensionIndex b) {
    if (a == dynamic_rank || b == dynamic_rank) return dynamic_rank;
    assert(a >= b);
    return RankConstraint(a - b);
  }

  /// Returns `true` if any rank satisfying `inner` also satisfies `outer`.
  ///
  /// \membergroup Comparison
  static constexpr bool Implies(DimensionIndex inner, DimensionIndex outer) {
    return outer == dynamic_rank || outer == inner;
  }

  /// Returns `true` if there is at least one rank satisfying all constraints.
  ///
  /// \membergroup Comparison
  static constexpr bool EqualOrUnspecified(DimensionIndex a, DimensionIndex b) {
    return a == dynamic_rank || b == dynamic_rank || a == b;
  }
  static constexpr bool EqualOrUnspecified(
      std::initializer_list<DimensionIndex> constraints) {
    DimensionIndex common = dynamic_rank;
    for (auto x : constraints) {
      if (x == dynamic_rank) continue;
      if (x != common && common != dynamic_rank) {
        return false;
      }
      common = x;
    }
    return true;
  }

  /// Returns `true` if some rank satisfying `a` is less than some rank
  /// satisfying `b`.
  ///
  /// \membergroup Comparison
  static constexpr bool LessOrUnspecified(DimensionIndex a, DimensionIndex b) {
    return a == dynamic_rank || b == dynamic_rank || a < b;
  }

  /// Returns `true` if some rank satisfying `a` is less than or equal to some
  /// rank satisfying `b`.
  ///
  /// \membergroup Comparison
  static constexpr bool LessEqualOrUnspecified(DimensionIndex a,
                                               DimensionIndex b) {
    return a == dynamic_rank || b == dynamic_rank || a <= b;
  }

  /// Returns `true` if some rank satisfying `a` is greater than some rank
  /// satisfying `b`.
  ///
  /// \membergroup Comparison
  static constexpr bool GreaterOrUnspecified(DimensionIndex a,
                                             DimensionIndex b) {
    return a == dynamic_rank || b == dynamic_rank || a > b;
  }

  /// Returns `true` if some rank satisfying `a` is greater than or equal to
  /// some rank satisfying `b`.
  ///
  /// \membergroup Comparison
  static constexpr bool GreaterEqualOrUnspecified(DimensionIndex a,
                                                  DimensionIndex b) {
    return a == dynamic_rank || b == dynamic_rank || a >= b;
  }
};

/// Checks if `inline_rank` is a valid compile-time rank constraint.
///
/// Supported values are:
///
/// - Any fixed rank for `0 <= inline_rank <= kMaxRank`.
///
/// - `dynamic_rank`, indicating an unspecified rank and no inline buffer.
///
/// - ``dynamic_rank(k)``, for ``0 <= k <= kMaxRank``, indicating an
///   unspecified rank with inline storage up to ``k``.
///
/// \relates InlineRank
constexpr inline bool IsValidInlineRank(InlineRank inline_rank) {
  return inline_rank >= (-kMaxRank - 1) && inline_rank <= kMaxRank;
}

/// Returns the inline rank limit of a rank spec.
///
/// \relates InlineRank
constexpr inline DimensionIndex InlineRankLimit(InlineRank rank_spec) {
  return (rank_spec <= -1) ? -1 - rank_spec : 0;
}

/// Template alias that evaluates to the type used for representing a static
/// rank.
///
/// This results in a substitution failure if `Rank < 0` (in particular, if
/// `Rank == dynamic_rank`).
///
/// If this is used to specify a default value for a function parameter, e.g.:
/// ``StaticOrDynamicRank<Rank> rank = StaticRank<Rank>()``, the effect is
/// that the parameter is required if `Rank == dynamic_rank`, but optional
/// otherwise.
///
/// \relates DimensionIndex
template <DimensionIndex Rank>
using StaticRank =
    std::enable_if_t<(Rank >= 0), std::integral_constant<DimensionIndex, Rank>>;

/// Template alias that evaluates to the type used for representing a static or
/// dynamic rank.
///
/// If `Rank == dynamic_rank`, this is `DimensionIndex`.  Otherwise, it is
/// `std::integral_constant<DimensionIndex, Rank>`.
///
/// \relates DimensionIndex
template <DimensionIndex Rank>
using StaticOrDynamicRank =
    std::conditional_t<(Rank <= dynamic_rank), DimensionIndex,
                       std::integral_constant<DimensionIndex, Rank>>;

/// If `Rank == dynamic_rank`, returns `dynamic_rank`.  Otherwise, returns
/// `StaticRank<Rank>{}`.
///
/// \relates RankConstraint
template <DimensionIndex Rank>
inline constexpr StaticOrDynamicRank<Rank> GetDefaultRank() {
  if constexpr (Rank == dynamic_rank) {
    return dynamic_rank;
  } else {
    return {};
  }
}

/// Evaluates to a type similar to `SourceRef` but with a static rank of
/// `TargetRank`.
///
/// Supported types include ``StaticOrDynamicRank<Rank>``
/// (i.e. `DimensionIndex` and
/// ``std::integral_constant<DimensionIndex, Rank>``), `Array`,
/// `StridedLayout`, `Box`, `TransformedArray`, `TensorStore`.
///
/// \tparam SourceRef Optionally ``const``- and/or reference-qualified source
///     type.  Any qualifiers are ignored.
/// \tparam TargetRank Target rank value.
/// \ingroup compile-time-constraints
template <typename SourceRef, DimensionIndex TargetRank>
using RebindRank =
    typename StaticCastTraitsType<SourceRef>::template RebindRank<TargetRank>;

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
/// Example::
///
///     StaticRank<3> r_static = StaticRankCast<3,
///     unchecked>(DimensionIndex(3)); Box<> box = ...; Result<Box<3>>
///     checked_result = StaticRankCast<3>(box); Box<3> unchecked_result =
///     StaticRankCast<3, unchecked>(box);
///
/// \tparam TargetRank Target rank value.
/// \tparam Checking Specifies whether the cast is checked or unchecked.
/// \param source Source value.
///
/// \ingroup compile-time-constraints
template <DimensionIndex TargetRank,
          CastChecking Checking = CastChecking::checked, typename SourceRef>
StaticCastResultType<RebindRank<SourceRef, TargetRank>, SourceRef, Checking>
StaticRankCast(SourceRef&& source) {
  return StaticCast<RebindRank<SourceRef, TargetRank>, Checking>(
      std::forward<SourceRef>(source));
}

// Specialization of `StaticCastTraits` for `DimensionIndex` (which is assumed
// to represent a rank value).
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

// Specialization of `StaticCastTraits` for
// `std::integral_constant<DimensionIndex, Rank>` (which is assumed to represent
// a rank value).
template <DimensionIndex Rank>
struct StaticCastTraits<std::integral_constant<DimensionIndex, Rank>> {
  static constexpr StaticRank<Rank> Construct(StaticRank<Rank>) { return {}; }
  static constexpr StaticRank<Rank> Construct(DimensionIndex) { return {}; }
  static constexpr bool IsCompatible(DimensionIndex source) {
    return RankConstraint::EqualOrUnspecified(source, Rank);
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

/// Validates that `0 <= rank <= kMaxRank`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `rank` is not valid.
/// \relates DimensionIndex
absl::Status ValidateRank(DimensionIndex rank);

}  // namespace tensorstore

#endif  // TENSORSTORE_RANK_H_
