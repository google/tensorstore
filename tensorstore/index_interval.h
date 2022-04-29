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

#ifndef TENSORSTORE_INDEX_INTERVAL_H_
#define TENSORSTORE_INDEX_INTERVAL_H_

#include <cassert>
#include <iosfwd>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/index.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Returns `true` if `index` is within the finite range:
/// [`kMinFiniteIndex`, `kMaxFiniteIndex`].
///
/// \relates Index
inline constexpr bool IsFiniteIndex(Index index) {
  return index >= kMinFiniteIndex && index <= kMaxFiniteIndex;
}

/// Return `true` if `index` is within the valid range:
/// [`-kInfIndex`, `+kInfIndex`].
///
/// \relates Index
inline constexpr bool IsValidIndex(Index index) {
  return index >= -kInfIndex && index <= +kInfIndex;
}

/// Represents an interval of index values, with support for +/-inf bounds.
///
/// \ingroup indexing
class IndexInterval {
 public:
  /// Constructs an interval corresponding to ``(-inf, +inf)``.
  constexpr IndexInterval() noexcept
      : inclusive_min_(-kInfIndex), size_(kInfSize) {}

  /// Returns an interval corresponding to ``(-inf, +inf)``.
  ///
  /// This is equivalent to the default constructor, but may be preferred for
  /// greater clarity.
  ///
  /// \membergroup Constructors
  constexpr static IndexInterval Infinite() noexcept { return {}; }

  /// Returns `true` if `inclusive_min` and `inclusive_max` specify a valid
  /// closed interval.
  ///
  /// \returns `inclusive_min >= -kInfIndex &&
  ///           inclusive_min < kInfIndex &&
  ///           inclusive_max > -kInfIndex &&
  ///           inclusive_max >= inclusive_min - 1 &&
  ///           inclusive_max <= kInfIndex`.
  constexpr static bool ValidClosed(Index inclusive_min, Index inclusive_max) {
    return inclusive_min >= -kInfIndex && inclusive_min < kInfIndex &&
           inclusive_max > -kInfIndex && inclusive_max >= inclusive_min - 1 &&
           inclusive_max <= kInfIndex;
  }

  /// Constructs an interval from inclusive lower and upper bounds.
  ///
  /// \pre `ValidClosed(inclusive_min, inclusive_max)`
  /// \returns `IndexInterval(inclusive_min, inclusive_max - inclusive_min + 1)`
  /// \membergroup Constructors
  static constexpr IndexInterval UncheckedClosed(Index inclusive_min,
                                                 Index inclusive_max) noexcept {
    assert(ValidClosed(inclusive_min, inclusive_max));
    return IndexInterval(inclusive_min, inclusive_max - inclusive_min + 1);
  }

  /// Tries to construct an interval from inclusive lower and upper bounds.
  ///
  /// \returns `UncheckedClosed(inclusive_min, inclusive_max)`
  /// \error `absl::StatusCode::kInvalidArgument` if
  ///     `!ValidClosed(inclusive_min, inclusive_max)`.
  /// \membergroup Constructors
  static Result<IndexInterval> Closed(Index inclusive_min, Index inclusive_max);

  /// Returns `true` if `inclusive_min` and `exclusive_max` specify a valid
  /// interval.
  ///
  /// \returns `inclusive_min >= -kInfIndex &&
  ///           inclusive_min < kInfIndex &&
  ///           exclusive_max > -kInfIndex + 1 &&
  ///           exclusive_max >= inclusive_min &&
  ///           exclusive_max <= kInfIndex + 1`.
  constexpr static bool ValidHalfOpen(Index inclusive_min,
                                      Index exclusive_max) {
    return inclusive_min >= -kInfIndex && inclusive_min < kInfIndex &&
           exclusive_max > -kInfIndex + 1 && exclusive_max >= inclusive_min &&
           exclusive_max <= kInfIndex + 1;
  }

  /// Constructs an interval from an inclusive lower bound and exclusive upper
  /// bound.
  ///
  /// \pre `ValidHalfOpen(inclusive_min, exclusive_max)`
  /// \returns `IndexInterval(inclusive_min, exclusive_max - inclusive_min)`.
  /// \membergroup Constructors
  static constexpr IndexInterval UncheckedHalfOpen(
      Index inclusive_min, Index exclusive_max) noexcept {
    assert(ValidHalfOpen(inclusive_min, exclusive_max));
    return IndexInterval(inclusive_min, exclusive_max - inclusive_min);
  }

  /// Tries to construct an interval from an inclusive lower bound and exclusive
  /// upper bound.
  ///
  /// \returns `UncheckedHalfOpen(inclusive_min, exclusive_max)`
  /// \error `absl::StatusCode::kInvalidArgument` if
  ///     `!ValidHalfOpen(inclusive_min, exclusive_max)`.
  /// \membergroup Constructors
  static Result<IndexInterval> HalfOpen(Index inclusive_min,
                                        Index exclusive_max);

  /// Returns `true` if `inclusive_min` and `size` specify a valid interval.
  ///
  /// \returns `inclusive_min >= -kInfIndex &&
  ///           size >= 0 &&
  ///           size <= kInfSize &&
  ///           inclusive_min < kInfIndex &&
  ///           inclusive_min <= kInfIndex + 1 - size &&
  ///           inclusive_min + size > -kInfIndex + 1`.
  constexpr static bool ValidSized(Index inclusive_min, Index size) {
    return inclusive_min >= -kInfIndex && size >= 0 && size <= kInfSize &&
           inclusive_min < kInfIndex && inclusive_min <= kInfIndex + 1 - size &&
           inclusive_min + size > -kInfIndex + 1;
  }

  /// Constructs an interval from the specified inclusive lower bound and size.
  ///
  /// \pre `ValidSized(inclusive_min, size)`.
  /// \membergroup Constructors
  static constexpr IndexInterval UncheckedSized(Index inclusive_min,
                                                Index size) {
    assert(ValidSized(inclusive_min, size));
    return IndexInterval(inclusive_min, size);
  }

  /// Tries to construct an interval from the specified inclusive lower bound
  /// and size.
  ///
  /// \returns `UncheckedSized(inclusive_min, size)`.
  /// \error `absl::StatusCode::kInvalidArgument` if `!ValidSized(inclusive_min,
  ///     size)`.
  /// \membergroup Constructors
  static Result<IndexInterval> Sized(Index inclusive_min, Index size);

  /// Returns the inclusive lower bound of the interval.
  ///
  /// \invariant inclusive_min() >= -kInfIndex
  /// \invariant inclusive_min() <= kMaxFiniteIndex
  /// \membergroup Accessors
  constexpr Index inclusive_min() const { return inclusive_min_; }

  /// Returns the exclusive lower bound.
  ///
  /// \returns inclusive_min() - 1
  /// \invariant exclusive_min() >= -kInfIndex - 1
  /// \membergroup Accessors
  constexpr Index exclusive_min() const { return inclusive_min_ - 1; }

  /// Returns the inclusive upper bound of the interval.
  ///
  /// \returns `inclusive_min() + size()`.
  /// \invariant exclusive_max() <= kInfIndex + 1
  /// \membergroup Accessors
  constexpr Index exclusive_max() const { return inclusive_min_ + size_; }

  /// Returns the inclusive upper bound of the interval.
  ///
  /// \returns `inclusive_min() + size() - 1`.
  /// \invariant inclusive_max() <= kInfIndex
  /// \invariant inclusive_max() >= kMinFiniteIndex
  /// \membergroup Accessors
  constexpr Index inclusive_max() const { return inclusive_min_ + size_ - 1; }

  /// Returns the size of the interval.
  ///
  /// If `inclusive_min() == -kInfIndex` and `inclusive_max() == +kInfIndex`,
  /// this returns `kInfSize` (as might be expected).  However, if
  /// `inclusive_min() == -kInfIndex` and `inclusive_max() < +kInfIndex`, or
  /// `inclusive_min() > -kInfIndex` and `inclusive_max() == +kInfIndex`, this
  /// returns a value less than `kInfSize`, even though the size is still
  /// logically unbounded.
  ///
  /// \invariant `size() >= 0 && size() <= kInfSize`
  /// \invariant `inclusive_min() + size() == exclusive_max()`
  /// \membergroup Accessors
  constexpr Index size() const { return size_; }

  /// Returns `size() == 0`.
  ///
  /// \membergroup Accessors
  constexpr bool empty() const { return size_ == 0; }

  // The following operators are defined as friends, even though they do not
  // require any special access to this class, in order to avoid slowing down
  // overload resolution.

  /// Writes a string representation of an interval to an ostream.
  ///
  /// The output format is "[lower, upper]", where lower may be "-inf" and upper
  /// may be "+inf".
  friend std::ostream& operator<<(std::ostream& os, IndexInterval x);

  /// Compares two intervals for equality.
  friend constexpr bool operator==(IndexInterval a, IndexInterval b) {
    return a.inclusive_min() == b.inclusive_min() && a.size() == b.size();
  }

  friend constexpr bool operator!=(IndexInterval a, IndexInterval b) {
    return !(a == b);
  }

  /// Returns the negated interval.
  ///
  /// Note that due to the constraints on `IndexInterval`, this operation cannot
  /// overflow.
  constexpr IndexInterval operator-() const {
    if (size_ == 0) return IndexInterval(-inclusive_min_, 0);
    return IndexInterval(-inclusive_max(), size());
  }

  template <typename H>
  friend H AbslHashValue(H h, IndexInterval x) {
    return H::combine(std::move(h), x.inclusive_min(), x.size());
  }

  /// Returns the full range of all valid finite index values.
  static constexpr IndexInterval FiniteRange() {
    return UncheckedClosed(kMinFiniteIndex, kMaxFiniteIndex);
  }

 private:
  explicit constexpr IndexInterval(Index inclusive_min, Index size) noexcept
      : inclusive_min_(inclusive_min), size_(size) {}
  friend class IndexIntervalRef;
  Index inclusive_min_;
  Index size_;
};

/// Returns `true` if `index` is contained within the `interval`.
///
/// \relates IndexInterval
/// \id interval, index
constexpr inline bool Contains(IndexInterval interval, Index index) {
  return index >= kMinFiniteIndex && index <= kMaxFiniteIndex &&
         index >= interval.inclusive_min() && index <= interval.inclusive_max();
}

/// Returns `true` if `outer` is a superset of `inner`.
///
/// \relates IndexInterval
/// \id interval, interval
constexpr inline bool Contains(IndexInterval outer, IndexInterval inner) {
  return inner.size() == 0 || (inner.inclusive_min() >= outer.inclusive_min() &&
                               inner.inclusive_max() <= outer.inclusive_max());
}

/// Returns `true` if `interval` is bounded below and above.
///
/// \relates IndexInterval
/// \id interval
constexpr inline bool IsFinite(IndexInterval interval) {
  return interval.inclusive_min() != -kInfIndex &&
         interval.inclusive_max() != kInfIndex;
}

/// Represents a mutable reference to an index interval stored as an
/// `inclusive_min`, `size` pair.
///
/// Assignment modifies the referenced `inclusive_min` and `size` values.
///
/// \relates IndexInterval
class IndexIntervalRef {
 public:
  constexpr explicit IndexIntervalRef(IndexInterval& other)  // NOLINT
      : IndexIntervalRef(other.inclusive_min_, other.size_) {}

  /// Converts to an `IndexInterval`.
  constexpr operator IndexInterval() const {
    return IndexInterval::UncheckedSized(inclusive_min(), size());
  }

  /// Assigns the referenced `inclusive_min` and `size` values.
  constexpr IndexIntervalRef& operator=(IndexInterval interval) noexcept {
    inclusive_min_ = interval.inclusive_min();
    size_ = interval.size();
    return *this;
  }
  constexpr IndexIntervalRef& operator=(IndexIntervalRef interval) noexcept {
    inclusive_min_ = interval.inclusive_min();
    size_ = interval.size();
    return *this;
  }

  /// Returns the inclusive lower bound.
  constexpr Index inclusive_min() const { return inclusive_min_; }

  /// Returns the size of the interval.
  constexpr Index size() const { return size_; }

  /// Returns `size() == 0`.
  constexpr bool empty() const { return size_ == 0; }

  /// Returns the exclusive lower bound.
  ///
  /// \returns `inclusive_min() - 1`.
  constexpr Index exclusive_min() const { return inclusive_min_ - 1; }

  /// Returns the inclusive upper bound of the interval.
  /// \returns `inclusive_min() + size()`.
  constexpr Index exclusive_max() const { return inclusive_min_ + size_; }

  /// Returns the inclusive upper bound of the interval.
  ///
  /// \returns `inclusive_min() + size() - 1`.
  constexpr Index inclusive_max() const { return inclusive_min_ + size_ - 1; }

  /// Returns an IndexIntervalRef that refers to the specified `inclusive_min`
  /// and `size` values.
  ///
  /// The values of `inclusive_min` and `size` are not checked at the time of
  /// construction.  However, any operation other than `operator=` is invalid if
  /// `IndexInterval::ValidSized(inclusive_min, size)` does not hold.
  static constexpr IndexIntervalRef UncheckedSized(
      Index& inclusive_min,  // NOLINT
      Index& size) {         // NOLINT
    return IndexIntervalRef(inclusive_min, size);
  }

  /// Prints a string representation.
  friend std::ostream& operator<<(std::ostream& os, IndexIntervalRef x) {
    return os << static_cast<IndexInterval>(x);
  }

 private:
  explicit constexpr IndexIntervalRef(Index& inclusive_min,  // NOLINT
                                      Index& size)           // NOLINT
      : inclusive_min_(inclusive_min), size_(size) {}

  Index& inclusive_min_;
  Index& size_;
};

/// Returns the smallest interval that contains `a` and `b`.
///
/// \relates IndexInterval
/// \id IndexInterval
IndexInterval Hull(IndexInterval a, IndexInterval b);

/// Returns the intersection of two intervals.
///
/// \relates IndexInterval
/// \id IndexInterval
IndexInterval Intersect(IndexInterval a, IndexInterval b);

/// Returns `Intersect(interval, IndexInterval::FiniteRange())`.
///
/// \relates IndexInterval
inline IndexInterval FiniteSubset(IndexInterval interval) {
  return Intersect(interval, IndexInterval::FiniteRange());
}

/// Returns `true` if the lower and upper bounds of `a` "match" the lower and
/// upper bounds of `b`.
///
/// An infinite bound is considered to match an infinite bound or any finite
/// bound, but two finite bounds only match if they are equal.
///
/// \relates IndexInterval
bool AreCompatibleOrUnbounded(IndexInterval a, IndexInterval b);

/// Returns `true` if the lower and upper bound of `inner` is either unbounded
/// or contained with the lower and upper bound, respectively, of `outer`.
///
/// \returns `(inner.inclusive_min() == -kInfIndex || inner.inclusive_min() >=
///     outer.inclusive_min()) && (inner.inclusive_max() == kInfIndex ||
///     inner.inclusive_max() <= outer.inclusive_max())`.
/// \relates IndexInterval
bool ContainsOrUnbounded(IndexInterval outer, IndexInterval inner);

/// Adds an offset to the min and max bounds of an interval.
///
/// If `interval.inclusive_min() == -kInfIndex`, it is not shifted.  Likewise,
/// if `interval.inclusive_max() == -kInfIndex`, it is also not shifted.
///
/// \param interval Original interval to shift.
/// \param min_offset Amount to add to `interval.inclusive_min()`.
/// \param max_offset Amount to add to `interval.inclusive_max()`.
/// \param offset Amount to add to `interval.inclusive_min()` and
///     `interval.inclusive_max()`.
/// \returns The shifted interval.
/// \error `absl::StatusCode::kInvalidArgument` if the resultant
///     ``inclusive_min`` or ``inclusive_max`` value would be outside the valid
///     range.
/// \relates IndexInterval
Result<IndexInterval> ShiftInterval(IndexInterval interval, Index min_offset,
                                    Index max_offset);
Result<IndexInterval> ShiftInterval(IndexInterval interval, Index offset);

/// Subtracts an offset from the min and max bounds of an interval.
///
/// Equivalent to `ShiftInterval(interval, -min_offset, -max_offset)`, except
/// that this function avoids overflow in the case that `min_offset` or
/// `max_offset` is equal to `std::numeric_limits<Index>::min()`.
///
/// \relates IndexInterval
Result<IndexInterval> ShiftIntervalBackward(IndexInterval interval,
                                            Index min_offset, Index max_offset);
Result<IndexInterval> ShiftIntervalBackward(IndexInterval interval,
                                            Index offset);

/// Shifts `interval.inclusive_min()` to `origin`.
///
/// The size is preserved, unless `interval.inclusive_min() == kInfIndex`.
///
/// \param interval The existing interval to shift.
/// \param origin The new ``inclusive_min`` value.
/// \returns The shifted interval.
/// \error `absl::StatusCode::kInvalidArgument` if `interval.inclusive_min() ==
///     -kInfIndex`. \error `absl::StatusCode::kOutOfRange` if `origin` is
///     outside ``[kMinFiniteIndex, kMaxFiniteIndex]``.
/// \error `absl::StatusCode::kInvalidArgument` if the resultant
///     ``inclusive_max`` value would be outside the valid range.
/// \relates IndexInterval
Result<IndexInterval> ShiftIntervalTo(IndexInterval interval, Index origin);

/// Checks that `interval.Contains(index)`.
///
/// \returns `absl::OkStatus()` on success.
/// \error `absl::StatusCode::kOutOfRange` on failure.
/// \relates IndexInterval
absl::Status CheckContains(IndexInterval interval, Index index);

/// Indicates how the interval bounds are specified to slicing operations.
///
/// \relates IndexInterval
enum class IntervalForm {
  /// Interval is specified by an inclusive lower bound and a size.
  sized,
  /// Interval is specified by inclusive lower and upper bounds.
  closed,
  /// Interval is specified by an inclusive lower bound and an exclusive upper
  /// bound.
  half_open,
};

/// Represents an IndexInterval where the lower/upper bounds may be "implicit".
///
/// When slicing, implicit bounds are used as the default bound if a
/// ``start``/``stop``/``size`` value of `kImplicit` is specified, but do not
/// constrain explicitly specified ``start``/``stop``/``size`` values.
///
/// \ingroup Indexing
class OptionallyImplicitIndexInterval : public IndexInterval {
 public:
  /// Constructs an infinite interval with implicit bounds.
  ///
  /// \id default
  constexpr OptionallyImplicitIndexInterval() noexcept = default;

  /// Constructs an interval with the specified bounds.
  ///
  /// \id interval, implicit
  constexpr OptionallyImplicitIndexInterval(IndexInterval interval,
                                            bool implicit_lower,
                                            bool implicit_upper) noexcept
      : IndexInterval(interval),
        implicit_lower_(implicit_lower),
        implicit_upper_(implicit_upper) {}

  /// Returns the base interval.
  ///
  /// \membergroup Accessors
  const IndexInterval& interval() const { return *this; }
  IndexInterval& interval() { return *this; }

  /// Indicates if the lower bound of `interval` is "implicit".
  ///
  /// \membergroup Accessors
  bool implicit_lower() const { return implicit_lower_; }
  bool& implicit_lower() { return implicit_lower_; }

  /// Indicates if the upper bound of `interval` is "implicit".
  ///
  /// \membergroup Accessors
  bool implicit_upper() const { return implicit_upper_; }
  bool& implicit_upper() { return implicit_upper_; }

  /// Returns the interval containing the effective bounds.
  ///
  /// The effective bounds are equal to `interval()`, except that an implicit
  /// lower/upper bound is converted to -/+inf, respectively.
  ///
  /// \membergroup Accessors
  IndexInterval effective_interval() const {
    return IndexInterval::UncheckedClosed(
        implicit_lower() ? -kInfIndex : inclusive_min(),
        implicit_upper() ? +kInfIndex : inclusive_max());
  }

  /// Prints a string representation of `x` to `os`.
  ///
  /// Implicit bounds are indicated with an asterisk, as in ``"[5, 10*]"`` (for
  /// an implicit upper bound), ``[5*, 10]`` (for an implicit lower bound), and
  /// ``[5*, 10*]`` (if both bounds are implicit).
  friend std::ostream& operator<<(std::ostream& os,
                                  const OptionallyImplicitIndexInterval& x);

  /// Compares two intervals for equality.
  friend bool operator==(const OptionallyImplicitIndexInterval& a,
                         const OptionallyImplicitIndexInterval& b) {
    return a.interval() == b.interval() &&
           a.implicit_lower() == b.implicit_lower() &&
           a.implicit_upper() == b.implicit_upper();
  }
  friend bool operator!=(const OptionallyImplicitIndexInterval& a,
                         const OptionallyImplicitIndexInterval& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const OptionallyImplicitIndexInterval& x) {
    return H::combine(std::move(h), x.interval(), x.implicit_lower(),
                      x.implicit_upper());
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.interval(), x.implicit_lower(), x.implicit_upper());
  };

 private:
  bool implicit_lower_ = true;
  bool implicit_upper_ = true;
};

/// Computes the hull of two index intervals.
///
/// Returns the smaller of the lower bounds and the larger of the upper bounds.
/// The ``implicit`` flag that corresponds to the selected bound is propagated,
/// in the event of equal bounds and mismatched implicit flags, an explicit flag
/// is used.
///
/// \param a OptionallyImplicitIndexInterval to hull.
/// \param b Other OptionallyImplicitIndexInterval to hull.
/// \relates OptionallyImplicitIndexInterval
/// \id OptionallyImplicitIndexInterval
OptionallyImplicitIndexInterval Hull(OptionallyImplicitIndexInterval a,
                                     OptionallyImplicitIndexInterval b);

/// Intersects two index intervals.
///
/// Returns the larger of the lower bounds and the smaller of the upper bounds.
/// The ``implicit`` flag that corresponds to the selected bound is propagated,
/// in the event of equal bounds and mismatched implicit flags, an explicit flag
/// is used.
///
/// \param a OptionallyImplicitIndexInterval to intersect.
/// \param b Other OptionallyImplicitIndexInterval to intersect.
/// \relates OptionallyImplicitIndexInterval
/// \id OptionallyImplicitIndexInterval
OptionallyImplicitIndexInterval Intersect(OptionallyImplicitIndexInterval a,
                                          OptionallyImplicitIndexInterval b);

/// Intersects two index intervals, preferring explicit bounds when implicit
/// flags mismatch.
///
/// Returns the larger of the lower bounds and the smaller of the upper bounds.
/// If the lower/upper bound is explicit in either a or b, then the lower/upper
/// bound of the result is explicit.
///
/// The normal intersection behavior applies if both bounds are either implicit
/// or explicit, but an explicit bound always overrides an implicit bound.
///
/// \param a OptionallyImplicitIndexInterval to intersect.
/// \param b Other OptionallyImplicitIndexInterval to intersect.
/// \relates OptionallyImplicitIndexInterval
OptionallyImplicitIndexInterval IntersectPreferringExplicit(
    OptionallyImplicitIndexInterval a, OptionallyImplicitIndexInterval b);

/// Represents an index interval with optionally-implicit bounds and an
/// optionally dimension label.
///
/// \ingroup Indexing
/// \tparam LabelCKind Specifies how the dimension label is stored.
///
///     - If `LabelCKind = container`, then the label is stored by value as an
///       `std::string`.
///
///     - If `LabelCKind = view`, the label is stored by reference as an
///       `std::string_view`.
template <ContainerKind LabelCKind = container>
class IndexDomainDimension : public OptionallyImplicitIndexInterval {
 public:
  /// Dimension label representation.
  using Label = std::conditional_t<LabelCKind == container, std::string,
                                   std::string_view>;

  /// Constructs an unlabeled dimension with infinite, implicit bounds.
  ///
  /// \id default
  IndexDomainDimension() = default;

  /// Constructs an unlabeled dimension with the specified bounds.
  ///
  /// \id interval
  IndexDomainDimension(const OptionallyImplicitIndexInterval& interval)
      : OptionallyImplicitIndexInterval(interval) {}

  /// Constructs a dimension with the given bounds and label.
  ///
  /// \id interval, label
  IndexDomainDimension(const OptionallyImplicitIndexInterval& interval,
                       Label label)
      : OptionallyImplicitIndexInterval(interval), label_(std::move(label)) {}

  /// Converts the label storage kind.
  ///
  /// \id convert
  template <ContainerKind OtherCKind>
  IndexDomainDimension(const IndexDomainDimension<OtherCKind>& other)
      : IndexDomainDimension(other.optionally_implicit_interval(),
                             Label(other.label())) {}

  /// Assigns the label and bounds.
  template <ContainerKind OtherCKind>
  IndexDomainDimension& operator=(
      const IndexDomainDimension<OtherCKind>& other) {
    optionally_implicit_interval() = other.optionally_implicit_interval();
    label_ = Label(other.label());
    return *this;
  }

  /// Returns the dimension bounds.
  const OptionallyImplicitIndexInterval& optionally_implicit_interval() const {
    return *this;
  }
  OptionallyImplicitIndexInterval& optionally_implicit_interval() {
    return *this;
  }

  /// Returns the dimension label.
  std::string_view label() const { return label_; }
  Label& label() { return label_; }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.optionally_implicit_interval(), x.label_);
  };

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
  friend std::ostream& operator<<(std::ostream& os,
                                  const IndexDomainDimension& x);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

  /// Compares the bounds and labels.
  friend bool operator==(const IndexDomainDimension<container>& a,
                         const IndexDomainDimension<container>& b);
  friend bool operator==(const IndexDomainDimension<view>& a,
                         const IndexDomainDimension<view>& b);
  friend bool operator==(const IndexDomainDimension<view>& a,
                         const IndexDomainDimension<container>& b);
  friend bool operator==(const IndexDomainDimension<container>& a,
                         const IndexDomainDimension<view>& b);

  template <ContainerKind OtherCKind>
  friend bool operator!=(const IndexDomainDimension<LabelCKind>& a,
                         const IndexDomainDimension<OtherCKind>& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const IndexDomainDimension& x) {
    return H::combine(std::move(h), x.optionally_implicit_interval(),
                      x.label());
  }

 private:
  Label label_;
};

// Prevent error-prone slicing comparison.
//
// Note: These are not declared as non-template friends within the
// IndexDomainDimension definition in order to work around Clang bug
// https://bugs.llvm.org/show_bug.cgi?id=35012

template <ContainerKind LabelCKind>
bool operator==(const IndexDomainDimension<LabelCKind>& a,
                const OptionallyImplicitIndexInterval& b) = delete;
template <ContainerKind LabelCKind>
bool operator!=(const IndexDomainDimension<LabelCKind>& a,
                const OptionallyImplicitIndexInterval& b) = delete;
template <ContainerKind LabelCKind>
bool operator==(const OptionallyImplicitIndexInterval& a,
                const IndexDomainDimension<LabelCKind>& b) = delete;
template <ContainerKind LabelCKind>
bool operator!=(const OptionallyImplicitIndexInterval& a,
                const IndexDomainDimension<LabelCKind>& b) = delete;

template <ContainerKind LabelCKind>
bool operator==(const IndexDomainDimension<LabelCKind>& a,
                const IndexInterval& b) = delete;
template <ContainerKind LabelCKind>
bool operator!=(const IndexDomainDimension<LabelCKind>& a,
                const IndexInterval& b) = delete;
template <ContainerKind LabelCKind>
bool operator==(const IndexInterval& a,
                const IndexDomainDimension<LabelCKind>& b) = delete;
template <ContainerKind LabelCKind>
bool operator!=(const IndexInterval& a,
                const IndexDomainDimension<LabelCKind>& b) = delete;

/// Merges two dimension labels.
///
/// The two labels are compatible if they are equal, or at most one is
/// non-empty.
///
/// The non-empty label takes precedence.
///
/// \param a Label to merge.
/// \param b Other label to merge.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
/// \relates IndexDomainDimension
Result<std::string_view> MergeDimensionLabels(std::string_view a,
                                              std::string_view b);

/// Merges two index intervals.
///
/// For both the lower and upper bounds, the bound of `a` and `b` must either be
/// equal (excluding the implicit indicator), or the bound in at least one of
/// `a` or `b` must be implicit and infinite.
///
/// The explicit/finite bounds take precedence over implicit/infinite bounds.
///
/// \param a Interval to merge.
/// \param b Other interval to merge.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
/// \relates OptionallyImplicitIndexInterval
Result<OptionallyImplicitIndexInterval> MergeOptionallyImplicitIndexIntervals(
    OptionallyImplicitIndexInterval a, OptionallyImplicitIndexInterval b);

/// Extracts a strided half-open interval from a containing interval.
///
/// This function is primarily for use by `DimExpression::HalfOpenInterval`.
///
/// The precise definition is as follows:
///
/// If `start == kImplicit`:
///
/// - Sets ``adjusted_start`` to ``orig.interval.inclusive_min()`` if
///   `stride > 0`, or ``orig.interval.inclusive_max()`` otherwise.
///
/// - Sets ``implicit_lower = orig.implicit_lower()``.
///
/// Otherwise (if `start != kImplicit`):
///
/// - Sets ``adjusted_start = start``.
/// - Sets ``implicit_lower = false``.
///
/// If `stop == kImplicit`:
///
/// - Sets ``adjusted_stop`` to ``orig.interval.inclusive_min()`` if
///   `stride < 0`, or ``orig.interval.inclusive_max()`` otherwise.
///
/// - Sets ``implicit_upper = orig.implicit_upper()``.
///
/// Otherwise (`stop != kImplicit`):
///
/// - Sets ``adjusted_stop = stop - sign(stride)``.
/// - Sets ``implicit_upper = false``.
///
/// If `stride > 0`:
///
/// - Sets ``adjusted_interval`` to ``[adjusted_start, adjusted_stop]``.
///
/// Otherwise (if `stride < 0`):
///
/// - Sets ``adjusted_interval`` to ``[adjusted_stop, adjusted_start]``.
/// - Swaps ``implicit_lower`` and ``implicit_upper``.
///
/// Sets ``new_inclusive_min = adjusted_start / stride`` (rounding towards
/// zero).
///
/// If ``adjusted_stop * sign(stride) == kInfIndex``:
///
/// -  Sets ``new_size = kInfIndex + 1 - new_inclusive_min``.
///
/// Otherwise:
///
/// - Sets ``new_size`` to the maximum positive integer such that
///   ``adjusted_start + stride * (new_size - 1)`` is contained in
///   ``adjusted_interval``.
///
/// Sets ``new_interval`` to be the interval starting at ``new_inclusive_min``
/// with a size of ``new_size``.
///
/// .. example:: Examples:
///
///   If ``orig = [5, 10]``, ``start = 6``, ``stop = 9``, and ``stride = 1``,
///   returns
///   ``[6, 8]`` with ``adjusted_start = 6``.
///
///   If ``orig = [5*, 10]``, ``start = 4``, ``stop = 9``, and ``stride = 1``,
///   returns
///   ``[4, 8]`` with ``adjusted_start = 4``.
///
///   If ``orig = [5*, 10]``, ``start = kImplicit``, ``stop = 9``, and ``stride
///   = 1``, returns ``[5*, 8]`` with ``adjusted_start = 5``.
///
///   If ``orig = [5, 10]``, ``start = 9``, ``stop = 7``, and ``stride = -2``,
///   returns ``[-4, -4]`` with ``adjusted_start = 9``.
///
/// \param orig The original interval from which to extract a strided
///     slice.
/// \param start The index in `orig` corresponding to the inclusive_min value in
///     the result interval.  If equal to `kImplicit`, the lower (if `stride >
///     0`) or upper (if `stride < 0`) bound of `orig` is used.
/// \param stop Specifies the exclusive stop index in `orig`.  If equal to
///     `kImplicit`, the upper (if `stride > 0`) or lower (if `stride < 0`)
///     bound of `orig` is used.
/// \param stride Specifies the stride value.
/// \returns
///     ``{{new_interval, implicit_lower, implicit_upper}, adjusted_start}``.
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if ``adjusted_interval`` is not
///     a valid interval.
/// \error `absl::StatusCode::kOutOfRange` if ``adjusted_interval`` is not
///     contained within `orig` (implicit bounds of `orig` do not constrain
///     ``adjusted_interval``).
/// \relates OptionallyImplicitIndexInterval
Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractHalfOpenStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                            Index stop, Index stride);

/// Extracts a strided closed interval from a containing interval.
///
/// This function is primarily for use by `DimExpression::ClosedInterval`.
///
/// The precise definition is as follows:
///
/// If `start == kImplicit`:
///
/// - Sets ``adjusted_start`` to ``orig.interval.inclusive_min()`` if
///   `stride > 0`, or ``orig.interval.inclusive_max()`` otherwise.
///
/// - Sets ``implicit_lower = orig.implicit_lower()``.
///
/// Otherwise (if `start != kImplicit`):
///
/// - Sets ``adjusted_start = start``.
/// - Sets ``implicit_lower = false``.
///
/// If `stop == kImplicit`:
///
/// - Sets ``adjusted_stop`` to ``orig.interval.inclusive_min()`` if
///   `stride < 0`, or ``orig.interval.inclusive_max()`` otherwise.
///
/// - Sets ``implicit_upper = orig.implicit_upper()``.
///
/// Otherwise (if `stop != kImplicit`):
///
/// - Sets ``adjusted_stop = stop``.
/// - Sets ``implicit_upper = false``.
///
/// If `stride > 0`:
///
/// - Sets ``adjusted_interval`` to ``[adjusted_start, adjusted_stop]``.
///
/// Otherwise (if `stride < 0`):
///
/// - Sets ``adjusted_interval`` to ``[adjusted_stop, adjusted_start]``.
/// - Swaps ``implicit_lower`` and ``implicit_upper``.
///
/// Sets ``new_inclusive_min = adjusted_start / stride`` (rounding towards
/// zero).
///
/// If ``adjusted_stop * sign(stride) == kInfIndex``:
///
/// - Sets ``new_size = kInfIndex + 1 - new_inclusive_min``.
///
/// Otherwise:
///
/// - Sets ``new_size`` to the maximum positive integer such that
///   ``adjusted_start + stride * (new_size - 1)`` is contained in
///   ``adjusted_interval``.
///
/// Sets ``new_interval`` to be the interval starting at
/// ``new_inclusive_min`` with a size of ``new_size``.
///
/// .. example::
///
///    If ``orig = [5, 10]``, ``start = 6``, ``stop = 9``, and ``stride = 1``,
///    returns ``[6, 9]`` with ``adjusted_start = 6``.
///
///    If ``orig = [5, 10]``, ``start = 9``, ``stop = 6``, and ``stride = -2``,
///    returns ``[-4, -3]`` with ``adjusted_start = 9``.
///
/// \param orig The original interval from which to extract a strided
///     slice.
/// \param start The index within `orig` corresponding to the inclusive_min
///     value in the result interval.  If equal to `kImplicit`, the lower (if
///     `stride > 0`) or upper (if `stride < 0`) bound of `orig` is used.
/// \param stop Specifies the inclusive stop index within `orig`.  If equal to
///     `kImplicit`, the upper (if `stride > 0`) or lower (if `stride < 0`)
///     bound of `orig` is used.
/// \param stride Specifies the stride value.
/// \returns
///     ``{{new_interval, implicit_lower, implicit_upper}, adjusted_start}``
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if ``adjusted_interval`` is not
///     a valid interval.
/// \error `absl::StatusCode::kOutOfRange` if ``adjusted_interval`` is not
///     contained within `orig` (implicit bounds of `orig` do not constrain
///     ``adjusted_interval``).
/// \relates OptionallyImplicitIndexInterval
Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractClosedStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                          Index stop, Index stride);

/// Extracts a strided interval of the specified size from a containing
/// interval.
///
/// This function is primarily for use by `DimExpression::SizedInterval`.
///
/// The precise definition is as follows:
///
/// If `start == kImplicit`:
///
/// - Sets ``adjusted_start`` to ``orig.interval.inclusive_min()`` if
///   `stride > 0`, or ``orig.interval.inclusive_max()`` otherwise.
///
/// - Sets ``implicit_lower = orig.implicit_lower()``.
///
/// Otherwise (if `start != kImplicit`):
///   Sets ``adjusted_start = start``.
///   Sets ``implicit_lower = false``.
///
/// Sets ``new_inclusive_min = adjusted_start / stride`` (rounding towards
/// zero).
///
/// If `size != kImplicit`:
///
/// - Sets ``new_size = size``.
///
/// Otherwise (if `size == kImplicit`):
///
/// - Sets ``new_size`` to the maximum positive integer such that
///   ``Contains(orig.interval, adjusted_start + stride * (new_size - 1))``,
///   or `0` if there is no such integer (can only occur if `orig.size() == 0`).
///
/// If `stride < 0`, swaps ``implicit_lower`` and ``implicit_upper``.
///
/// Sets ``new_interval`` to be the interval starting at ``new_inclusive_min``
/// with a size of ``new_size``.
///
/// .. example:: Examples
///
///   If ``orig = [5, 10]``, ``start = 9``, ``stop_or_size = 3``, and ``stride =
///   -2``, returns ``[-4, -2]`` with ``adjusted_start = 9``.
///
/// \param orig The original interval from which to extract a strided
///     slice.
/// \param start The index within `orig` corresponding to the inclusive_min
///     value in the result interval.  If equal to `kImplicit`, the lower (if
///     `stride > 0`) or upper (if `stride < 0`) bound of `orig` is used.
/// \param size Specifies the size of the result interval.
/// \param stride Specifies the stride value.
/// \returns
///     ``{{new_interval, implicit_lower, implicit_upper}, adjusted_start}``.
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if `size < 0`.
/// \error `absl::StatusCode::kOutOfRange` if ``new_size > 0`` and `orig`
///     does not contain ``adjusted_start + stride * (new_size - 1)``
///     (implicit bounds of `orig` are not constraints).
/// \relates IndexInterval
/// \relates OptionallyImplicitIndexInterval
Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractSizedStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                         Index size, Index stride);

/// Equivalent to `ExtractHalfOpenStridedSlice`, `ExtractHalfOpenStridedSlice`,
/// or `ExtractHalfOpenStridedSlice` depending on the value of `interval_form`.
///
/// \relates OptionallyImplicitIndexInterval
Result<std::pair<OptionallyImplicitIndexInterval, Index>> ExtractStridedSlice(
    OptionallyImplicitIndexInterval orig, IntervalForm interval_form,
    Index start, Index stop_or_size, Index stride);

/// Computes a mapping from the specified interval back to the original domain.
///
/// \param orig The original domain.
/// \param interval_form Form of the interval.
/// \param translate_origin_to If not equal to `kImplicit`, the resultant
///     `*new_domain` is translated to the specified origin.
/// \param start The index within `orig` corresponding to the ``inclusive_min``
///     in the resultant value of `*new_domain`.  If equal to `kImplicit`, the
///     lower (if `stride > 0`) or upper (if `stride < 0`) bound of `orig` is
///     used.
/// \param stop_or_size Specifies the inclusive/exclusive stop index or size of
///     the resultant `*new_domain`.
/// \param stride Specifies the stride value.
/// \param new_domain[out] Non-null pointer to result interval set to the new
///     domain computed by `ExtractStridedSlice`, except that it is translated
///     according to the value of `translate_origin_to`.
/// \param output_offset[out] Non-null pointer set to the value such that
///     ``*output_offset + stride * i`` maps each value ``i`` in the resultant
///     `*new_domain` to the corresponding index in `orig`.
/// \error `absl::StatusCode::kInvalidArgument` or
///     `absl::StatusCode::kOutOfRange` if the slice is not valid.
/// \relates OptionallyImplicitIndexInterval
absl::Status ComputeStridedSliceMap(OptionallyImplicitIndexInterval orig,
                                    IntervalForm interval_form,
                                    Index translate_origin_to, Index start,
                                    Index stop_or_size, Index stride,
                                    OptionallyImplicitIndexInterval* new_domain,
                                    Index* output_offset);

/// Computes the largest possible domain such that the affine-transformed range
/// is contained within `interval`.
///
/// Specifically, subtracts `offset` from the lower and upper bounds of
/// `interval` and then divides the bounds by `divisor`, rounding towards the
/// interior of the interval.
///
/// A bound of `-kInfIndex` or `kInfIndex` is not affected by subtraction or
/// division.
///
/// \dchecks `divisor != 0`
/// \param interval The range of the affine transform.
/// \param offset The offset of the affine transform.
/// \param divisor The multiplier of the affine transform.
/// \returns The domain interval.
/// \error `absl::StatusCode::kInvalidArgument` if the domain would overflow.
/// \relates IndexInterval
Result<IndexInterval> GetAffineTransformDomain(IndexInterval interval,
                                               Index offset, Index divisor);

/// Same as above, but also propagates `interval.implicit_lower()` and
/// `interval.implicit_upper()` to the input space.
///
/// \relates IndexInterval
/// \id OptionallyImplicitIndexInterval
Result<OptionallyImplicitIndexInterval> GetAffineTransformDomain(
    OptionallyImplicitIndexInterval interval, Index offset, Index divisor);

/// Computes the range of the affine-transformed domain `interval`.
///
/// \param interval The domain to be transformed.
/// \param offset The offset by which to shift the multiplied interval.
/// \param multiplier The multiplier by which to multiply `interval`.
/// \error `absl::StatusCode::kInvalidArgument` if the result interval cannot be
///     represented.
/// \relates IndexInterval
/// \id IndexInterval
Result<IndexInterval> GetAffineTransformRange(IndexInterval interval,
                                              Index offset, Index multiplier);

/// Same as above, but also propagates `interval.implicit_lower()` and
/// `interval.implicit_upper()` to the output space.
///
/// \relates OptionallyImplicitIndexInterval
/// \id OptionallyImplicitIndexInterval
Result<OptionallyImplicitIndexInterval> GetAffineTransformRange(
    OptionallyImplicitIndexInterval interval, Index offset, Index multiplier);

/// Computes the interval containing all indices ``x`` for which ``(x - offset)
/// / divisor`` is in `interval`, where ``/`` rounds towards 0.
///
/// The result is the same as `GetAffineTransformRange`, except that for
/// non-empty `interval`:
///
/// - if `divisor > 0`: the upper bound is expanded by `divisor - 1`;
///
/// - if `divisor < 0`: the lower bound is expanded by `-divisor - 1`.
///
/// .. example:: For example:
///
///     GetAffineTransformRange([2, 4], 1, 3)          -> [  7, 13]
///     GetAffineTransformInverseDomain([2, 4], 1, 3)  -> [  7, 15]
///     GetAffineTransformRange([2, 4], 1, -3)         -> [-11, -5]
///     GetAffineTransformInverseDomain([2, 4], 1, -3) -> [-13, -5]
///
/// \param interval The domain of the affine transform.
/// \param offset The offset of the affine transform.
/// \param divisor The multiplier of the affine transform.
/// \returns The range of the affine transform.
/// \error `absl::StatusCode::kInvalidArgument` if the result interval cannot be
///     represented.
/// \relates IndexInterval
Result<IndexInterval> GetAffineTransformInverseDomain(IndexInterval interval,
                                                      Index offset,
                                                      Index divisor);

/// Returns `index` if `index != kImplicit`, or `default_value` otherwise.
///
/// \relates Index
constexpr inline Index ExplicitIndexOr(Index index, Index default_value) {
  return index == kImplicit ? default_value : index;
}

/// Returns `true` if `index` is either `kImplicit` or `expected`.
///
/// \relates Index
constexpr inline bool ImplicitOrEqual(Index index, Index expected) {
  return index == kImplicit || index == expected;
}

/// Divides the lower and upper bounds of `interval` by `divisor`, rounding out
/// (expanding the interval) to the nearest integer.
///
/// \dchecks `divisor > 0`
/// \relates IndexInterval
constexpr inline IndexInterval DividePositiveRoundOut(IndexInterval interval,
                                                      Index divisor) {
  assert(divisor > 0);
  return IndexInterval::UncheckedHalfOpen(
      FloorOfRatio(interval.inclusive_min(), divisor),
      CeilOfRatio(interval.exclusive_max(), divisor));
}

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::IndexInterval)

#endif  // TENSORSTORE_INDEX_INTERVAL_H_
