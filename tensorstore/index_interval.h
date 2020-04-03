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

#include <iosfwd>
#include <utility>

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/index.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Returns `true` if `index` is within the finite range:
/// `[kMinFiniteIndex, kMaxFiniteIndex]`.
inline constexpr bool IsFiniteIndex(Index index) {
  return index >= kMinFiniteIndex && index <= kMaxFiniteIndex;
}

/// Return `true` if `index` is within the valid range:
/// `[-kInfIndex, +kInfIndex]`.
inline constexpr bool IsValidIndex(Index index) {
  return index >= -kInfIndex && index <= +kInfIndex;
}

/// Represents an interval of index values, with support for +/-inf bounds.
class IndexInterval {
 public:
  /// Constructs an interval corresponding to [-inf, +inf].
  constexpr IndexInterval() noexcept
      : inclusive_min_(-kInfIndex), size_(kInfSize) {}

  /// Returns `true` if `inclusive_min` and `inclusive_max` specify a valid
  /// interval.
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
  static Result<IndexInterval> Sized(Index inclusive_min, Index size);

  /// Returns the inclusive lower bound of the interval.
  /// \invariant inclusive_min() >= -kInfIndex
  /// \invariant inclusive_min() <= kMaxFiniteIndex
  constexpr Index inclusive_min() const { return inclusive_min_; }

  /// Returns the exclusive lower bound.
  /// \returns inclusive_min() - 1
  /// \invariant exclusive_min() >= -kInfIndex - 1
  constexpr Index exclusive_min() const { return inclusive_min_ - 1; }

  /// Returns the inclusive upper bound of the interval.
  /// \returns `inclusive_min() + size()`.
  /// \invariant exclusive_max() <= kInfIndex + 1
  constexpr Index exclusive_max() const { return inclusive_min_ + size_; }

  /// Returns the inclusive upper bound of the interval.
  /// \returns `inclusive_min() + size() - 1`.
  /// \invariant inclusive_max() <= kInfIndex
  /// \invariant inclusive_max() >= kMinFiniteIndex
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
  constexpr Index size() const { return size_; }

  /// Returns `size() == 0`.
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

  template <typename H>
  friend H AbslHashValue(H h, IndexInterval x) {
    return H::combine(std::move(h), x.inclusive_min(), x.size());
  }

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
constexpr inline bool Contains(IndexInterval interval, Index index) {
  return index >= kMinFiniteIndex && index <= kMaxFiniteIndex &&
         index >= interval.inclusive_min() && index <= interval.inclusive_max();
}

/// Returns `true` if `outer` is a superset of `inner`.
constexpr inline bool Contains(IndexInterval outer, IndexInterval inner) {
  return inner.size() == 0 || (inner.inclusive_min() >= outer.inclusive_min() &&
                               inner.inclusive_max() <= outer.inclusive_max());
}

/// Returns `true` if `interval` is bounded below and above.
constexpr inline bool IsFinite(IndexInterval interval) {
  return interval.inclusive_min() != -kInfIndex &&
         interval.inclusive_max() != kInfIndex;
}

/// Represents a mutable reference to an index interval stored as an
/// `inclusive_min`, `size` pair.
///
/// Assignment modifies the referenced `inclusive_min` and `size` values.
class IndexIntervalRef {
 public:
  constexpr explicit IndexIntervalRef(IndexInterval& other)  // NOLINT
      : IndexIntervalRef(other.inclusive_min_, other.size_) {}

  constexpr operator IndexInterval() const {
    return IndexInterval::UncheckedSized(inclusive_min(), size());
  }

  /// Assigns the referenced `inclusive_min` and `size` values.
  constexpr IndexIntervalRef& operator=(IndexInterval interval) noexcept {
    inclusive_min_ = interval.inclusive_min();
    size_ = interval.size();
    return *this;
  }

  /// Assigns the referenced `inclusive_min` and `size` values.
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
  /// \returns `inclusive_min() - 1`.
  constexpr Index exclusive_min() const { return inclusive_min_ - 1; }

  /// Returns the inclusive upper bound of the interval.
  /// \returns `inclusive_min() + size()`.
  constexpr Index exclusive_max() const { return inclusive_min_ + size_; }

  /// Returns the inclusive upper bound of the interval.
  /// \returns `inclusive_min() + size() - 1`.
  constexpr Index inclusive_max() const { return inclusive_min_ + size_ - 1; }

  /// Returns an IndexIntervalRef that refers to the specified `inclusive_min`
  /// and `size` values.
  ///
  /// The values of `inclusive_min` and `sized` are not checked at the time of
  /// construction.  However, any operation other than `operator=` is invalid if
  /// `IndexInterval::ValidSized(inclusive_min, size)` does not hold.
  static constexpr IndexIntervalRef UncheckedSized(
      Index& inclusive_min,  // NOLINT
      Index& size) {         // NOLINT
    return IndexIntervalRef(inclusive_min, size);
  }

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

/// Returns the intersection of two intervals.
IndexInterval Intersect(IndexInterval a, IndexInterval b);

/// Returns the smallest interval that contains `a` and `b`.
IndexInterval Hull(IndexInterval a, IndexInterval b);

/// Returns `Intersect(interval, IndexInterval::FiniteRange())`.
inline IndexInterval FiniteSubset(IndexInterval interval) {
  return Intersect(interval, IndexInterval::FiniteRange());
}

/// Returns `true` if the lower and upper bounds of `a` "match" the lower and
/// upper bounds of `b`.
///
/// An infinite bound is considered to match an infinite bound or any finite
/// bound, but two finite bounds only match if they are equal.
bool AreCompatibleOrUnbounded(IndexInterval a, IndexInterval b);

/// Returns `true` if the lower and upper bound of `inner` is either unbounded
/// or contained with the lower and upper bound, respectively, of `outer`.
///
/// \returns `(inner.inclusive_min() == -kInfIndex || inner.inclusive_min() >=
///     outer.inclusive_min()) && (inner.inclusive_max() == kInfIndex ||
///     inner.inclusive_max() <= outer.inclusive_max())`.
bool ContainsOrUnbounded(IndexInterval outer, IndexInterval inner);

/// Adds an offset to both the min and max bounds of an interval.
///
/// If `interval.inclusive_min() == -kInfIndex`, it is not shifted.  Likewise,
/// if `interval.inclusive_max() == -kInfIndex`, it is also not shifted.
///
/// \returns The shifted interval.
/// \error `absl::StatusCode::kOutOfRange` if `offset < kMinFiniteIndex` or
///     `offset > kMaxFiniteIndex`.
/// \error `absl::StatusCode::kInvalidArgument` if the resultant `inclusive_min`
///     or `inclusive_max` value would be outside the valid range.
Result<IndexInterval> ShiftInterval(IndexInterval interval, Index offset);

/// Shifts the `inclusive_min` value of `interval` to `origin`.
///
/// The size is preserved, unless `interval.inclusive_min() == kInfIndex`.
///
/// \param interval The existing interval to shift.
/// \param origin The new `inclusive_min` value.
/// \returns The shifted interval.
/// \error `absl::StatusCode::kInvalidArgument` if `interval.inclusive_min() ==
///     -kInfIndex`. \error `absl::StatusCode::kOutOfRange` if `origin` is
///     outside `[kMinFiniteIndex, kMaxFiniteIndex]`.
/// \error `absl::StatusCode::kInvalidArgument` if the resultant `inclusive_max`
///     value would be outside the valid range.
Result<IndexInterval> ShiftIntervalTo(IndexInterval interval, Index origin);

/// Checks that `interval.Contains(index)`.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kOutOfRange` on failure.
Status CheckContains(IndexInterval interval, Index index);

enum class IntervalForm { sized, closed, half_open };

/// Represents an IndexInterval where the lower/upper bounds may be "implicit".
///
/// When slicing, implicit bounds are used as the default bound if a
/// `start`/`stop`/`size` value of `kImplicit` is specified, but do not
/// constrain explicitly specified `start`/`stop`/`size` values.
class OptionallyImplicitIndexInterval : public IndexInterval {
 public:
  constexpr OptionallyImplicitIndexInterval() noexcept = default;

  constexpr OptionallyImplicitIndexInterval(IndexInterval interval,
                                            bool implicit_lower,
                                            bool implicit_upper) noexcept
      : IndexInterval(interval),
        implicit_lower_(implicit_lower),
        implicit_upper_(implicit_upper) {}

  /// Prints a string representation of `x` to `os`.
  ///
  /// Implicit bounds are indicated with an asterisk, as in `"[5, 10*]"` (for an
  /// implicit upper bound), `[5*, 10]` (for an implicit lower bound), and
  /// `[5*, 10*]` (if both bounds are implicit).
  friend std::ostream& operator<<(std::ostream& os,
                                  const OptionallyImplicitIndexInterval& x);

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

  const IndexInterval& interval() const { return *this; }
  IndexInterval& interval() { return *this; }

  /// Indicates if the lower bound of `interval` is "implicit".
  bool implicit_lower() const { return implicit_lower_; }
  bool& implicit_lower() { return implicit_lower_; }

  /// Indicates if the upper bound of `interval` is "implicit".
  bool implicit_upper() const { return implicit_upper_; }
  bool& implicit_upper() { return implicit_upper_; }

  /// Returns the interval containing the effective bounds.
  ///
  /// The effective bounds are equal to `interval()`, except that an implicit
  /// lower/upper bound is converted to -/+inf, respectively.
  IndexInterval effective_interval() const {
    return IndexInterval::UncheckedClosed(
        implicit_lower() ? -kInfIndex : inclusive_min(),
        implicit_upper() ? +kInfIndex : inclusive_max());
  }

 private:
  bool implicit_lower_ = true;
  bool implicit_upper_ = true;
};

/// Returns the intersection of two intervals.
///
/// The normal intersection behavior applies if both bounds are either implicit
/// or explicit, but an explicit bound always overrides an implicit bound.
OptionallyImplicitIndexInterval Intersect(OptionallyImplicitIndexInterval a,
                                          OptionallyImplicitIndexInterval b);

/// Represents an index interval with optionally-implicit bounds and an
/// optionally dimension label.
template <ContainerKind LabelCKind = container>
class IndexDomainDimension : public OptionallyImplicitIndexInterval {
 public:
  using Label = absl::conditional_t<LabelCKind == container, std::string,
                                    absl::string_view>;
  IndexDomainDimension() = default;

  IndexDomainDimension(const OptionallyImplicitIndexInterval& interval)
      : OptionallyImplicitIndexInterval(interval) {}

  IndexDomainDimension(const OptionallyImplicitIndexInterval& interval,
                       Label label)
      : OptionallyImplicitIndexInterval(interval), label_(std::move(label)) {}

  template <ContainerKind OtherCKind>
  IndexDomainDimension(const IndexDomainDimension<OtherCKind>& other)
      : IndexDomainDimension(other.optionally_implicit_interval(),
                             Label(other.label())) {}

  template <ContainerKind OtherCKind>
  IndexDomainDimension& operator=(
      const IndexDomainDimension<OtherCKind>& other) {
    optionally_implicit_interval() = other.optionally_implicit_interval();
    label_ = Label(other.label());
    return *this;
  }

  const OptionallyImplicitIndexInterval& optionally_implicit_interval() const {
    return *this;
  }
  OptionallyImplicitIndexInterval& optionally_implicit_interval() {
    return *this;
  }

  absl::string_view label() const { return label_; }
  Label& label() { return label_; }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
  friend std::ostream& operator<<(std::ostream& os,
                                  const IndexDomainDimension& x);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

/// Extracts a strided half-open interval from a containing interval.
///
/// This function is primarily for use by `DimExpression::HalfOpenInterval`.
///
/// The precise definition is as follows:
///
/// If `start == kImplicit`:
///   Sets `adjusted_start = stride > 0 ? orig.interval.inclusive_min()
///                                     : orig.interval.inclusive_max()`.
///   Sets `implicit_lower = orig.implicit_lower`.
/// Otherwise (if `start != kImplicit`):
///   Sets `adjusted_start = start`.
///   Sets `implicit_lower = false`.
///
/// If `stop == kImplicit`:
///   Sets `adjusted_stop = stride < 0 ? orig.interval.inclusive_min()
///                                    : orig.interval.inclusive_max()`.
///   Sets `implicit_upper = orig.implicit_upper`.
/// Otherwise (`stop != kImplicit`):
///   Sets `adjusted_stop = stop - sign(stride)`.
///   Sets `implicit_upper = false`.
///
/// If `stride > 0`:
///   Sets `adjusted_interval` to `[adjusted_start, adjusted_stop]`.
/// Otherwise (if `stride < 0 `):
///   Sets `adjusted_interval` to `[adjusted_stop, adjusted_start]`.
///   Swaps `implicit_lower` and `implicit_upper`.
///
/// Sets `new_inclusive_min = adjusted_start / stride` (rounding towards zero).
///
/// If `adjusted_stop * sign(stride) == kInfIndex`:
///   Sets `new_size = kInfIndex + 1 - new_inclusive_min`.
/// Otherwise:
///   Sets `new_size` to the maximum positive integer such that
///   `adjusted_start + stride * (new_size - 1)` is contained in
///   `adjusted_interval`.
///
/// Sets `new_interval` to be the interval starting at `new_inclusive_min` with
/// a size of `new_size`.
///
/// Examples:
///
///   If `orig = [5, 10]`, `start = 6`, `stop = 9`, and `stride = 1`, returns
///   `[6, 8]` with `adjusted_start = 6`.
///
///   If `orig = [5*, 10]`, `start = 4`, `stop = 9`, and `stride = 1`, returns
///   `[4, 8]` with `adjusted_start = 4`.
///
///   If `orig = [5*, 10]`, `start = kImplicit`, `stop = 9`, and `stride = 1`,
///   returns `[5*, 8]` with `adjusted_start = 5`.
///
///   If `orig = [5, 10]`, `start = 9`, `stop = 7`, and `stride = -2`,
///   returns `[-4, -4]` with `adjusted_start = 9`.
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
/// \returns `{{new_interval, implicit_lower, implicit_upper}, adjusted_start}`.
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if `adjusted_interval` is not a
///     valid interval.
/// \error `absl::StatusCode::kOutOfRange` if `adjusted_interval` is not
///     contained within `orig` (implicit bounds of `orig` do not constrain
///     `adjusted_interval`).
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
///   Sets `adjusted_start = stride > 0 ? orig.interval.inclusive_min()
///                                     : orig.interval.inclusive_max()`.
///   Sets `implicit_lower = orig.implicit_lower`.
/// Otherwise (if `start != kImplicit`):
///   Sets `adjusted_start = start`.
///   Sets `implicit_lower = false`.
///
/// If `stop == kImplicit`:
///   Sets `adjusted_stop = stride < 0 ? orig.interval.inclusive_min()
///                                    : orig.interval.inclusive_max()`.
///   Sets `implicit_upper = orig.implicit_upper`.
/// Otherwise (if `stop != kImplicit`):
///   Sets `adjusted_stop = stop`.
///   Sets `implicit_upper = false`.
///
/// If `stride > 0`:
///   Sets `adjusted_interval` to `[adjusted_start, adjusted_stop]`.
/// Otherwise (if `stride < 0 `):
///   Sets `adjusted_interval` to `[adjusted_stop, adjusted_start]`.
///   Swaps `implicit_lower` and `implicit_upper`.
///
/// Sets `new_inclusive_min = adjusted_start / stride` (rounding towards zero).
///
/// If `adjusted_stop * sign(stride) == kInfIndex`:
///   Sets `new_size = kInfIndex + 1 - new_inclusive_min`.
/// Otherwise:
///   Sets `new_size` to the maximum positive integer such that
///   `adjusted_start + stride * (new_size - 1)` is contained in
///   `adjusted_interval`.
///
/// Sets `new_interval` to be the interval starting at `new_inclusive_min` with
/// a size of `new_size`.
///
/// Examples:
///
///   If `orig = [5, 10]`, `start = 6`, `stop = 9`, and `stride = 1`,
///   returns `[6, 9]` with `adjusted_start = 6`.
///
///   If `orig = [5, 10]`, `start = 9`, `stop = 6`, and `stride = -2`,
///   returns `[-4, -3]` with `adjusted_start = 9`.
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
/// \returns `{{new_interval, implicit_lower, implicit_upper}, adjusted_start}`.
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if `adjusted_interval` is not a
///     valid interval.
/// \error `absl::StatusCode::kOutOfRange` if `adjusted_interval` is not
///     contained within `orig` (implicit bounds of `orig` do not constrain
///     `adjusted_interval`).
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
///   Sets `adjusted_start = stride > 0 ? orig.interval.inclusive_min()
///                                     : orig.interval.inclusive_max()`.
///   Sets `implicit_lower = orig.implicit_lower`.
/// Otherwise (if `start != kImplicit`):
///   Sets `adjusted_start = start`.
///   Sets `implicit_lower = false`.
///
/// Sets `new_inclusive_min = adjusted_start / stride` (rounding towards zero).
///
/// If `size != kImplicit`:
///   Sets `new_size = size`.
/// Otherwise (if `size == kImplicit`):
///   Sets `new_size` to the maximum positive integer such that
///   `orig.interval.Contains(adjusted_start + stride * (new_size - 1))`, or `0`
///   if there is no such integer (can only occur if `orig.size() == 0`).
///
/// If `stride < 0 `, swaps `implicit_lower` and `implicit_upper`.
///
/// Sets `new_interval` to be the interval starting at `new_inclusive_min` with
/// a size of `new_size`.
///
/// Examples:
///
///   If `orig = [5, 10]`, `start = 9`, `stop_or_size = 3`, and `stride = -2`,
///   returns `[-4, -2]` with `adjusted_start = 9`.
///
/// \param orig The original interval from which to extract a strided
///     slice.
/// \param start The index within `orig` corresponding to the inclusive_min
///     value in the result interval.  If equal to `kImplicit`, the lower (if
///     `stride > 0`) or upper (if `stride < 0`) bound of `orig` is used.
/// \param size Specifies the size of the result interval.
/// \param stride Specifies the stride value.
/// \returns `{{new_interval, implicit_lower, implicit_upper}, adjusted_start}`.
/// \error `absl::StatusCode::kInvalidArgument` if `stride == 0` or
///     `stride == std::numeric_limits<Index>::min()`.
/// \error `absl::StatusCode::kInvalidArgument` if `size < 0`.
/// \error `absl::StatusCode::kOutOfRange` if `new_size > 0` and `orig` does not
///     contain `adjusted_start + stride * (new_size - 1)` (implicit bounds of
///     `orig` are not constraints).
Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractSizedStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                         Index size, Index stride);

/// Equivalent to `ExtractHalfOpenStridedSlice`, `ExtractHalfOpenStridedSlice`,
/// or `ExtractHalfOpenStridedSlice` depending on the value of `interval_form`.
Result<std::pair<OptionallyImplicitIndexInterval, Index>> ExtractStridedSlice(
    OptionallyImplicitIndexInterval orig, IntervalForm interval_form,
    Index start, Index stop_or_size, Index stride);

/// Computes a mapping from the specified interval back to the original domain.
///
/// \param orig The original domain.
/// \param interval_form Form of the interval.
/// \param translate_origin_to If not equal to `kImplicit`, the resultant
///     `*new_domain` is translated to the specified origin.
/// \param start The index within `orig` corresponding to the `inclusive_min` in
///     the resultant value of `*new_domain`.  If equal to `kImplicit`, the
///     lower (if `stride > 0`) or upper (if `stride < 0`) bound of `orig` is
///     used.
/// \param stop_or_size Specifies the inclusive/exclusive stop index or size of
///     the resultant `*new_domain`.
/// \param stride Specifies the stride value.
/// \param new_domain[out] Non-null pointer to result interval set to the new
///     domain computed by `ExtractStridedSlice`, except that it is translated
///     according to the value of `translate_origin_to`.
/// \param output_offset[out] Non-null pointer set to the value such that
///     `*output_offset + stride * i` maps each value `i` in the resultant
///     `*new_domain` to the corresponding index in `orig`.
/// \error `absl::StatusCode:kInvalidArgument` or
///     `absl::StatusCode::kOutOfRange` if the slice is not valid.
Status ComputeStridedSliceMap(OptionallyImplicitIndexInterval orig,
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
Result<IndexInterval> GetAffineTransformDomain(IndexInterval interval,
                                               Index offset, Index divisor);

/// Same as above, but also propagates `implicit_lower` and `implicit_upper` to
/// the input space.
Result<OptionallyImplicitIndexInterval> GetAffineTransformDomain(
    OptionallyImplicitIndexInterval interval, Index offset, Index divisor);

/// Computes the range of the affine-transformed domain `interval`.
///
/// \param interval The domain to be transformed.
/// \param offset The offset by which to shift the multiplied interval.
/// \param multiplier The multiplier by which to multiply `interval`.
/// \error `absl::StatusCode::kInvalidArgument` if the result interval cannot be
///     represented.
Result<IndexInterval> GetAffineTransformRange(IndexInterval interval,
                                              Index offset, Index multiplier);

/// Same as above, but also propagates `implicit_lower` and `implicit_upper` to
/// the output space.
Result<OptionallyImplicitIndexInterval> GetAffineTransformRange(
    OptionallyImplicitIndexInterval interval, Index offset, Index multiplier);

/// Returns `index` if `index != kImplicit`, or `default_value` otherwise.
constexpr inline Index ExplicitIndexOr(Index index, Index default_value) {
  return index == kImplicit ? default_value : index;
}

/// Returns `true` if `index` is either `kImplicit` or `expected`.
constexpr inline bool ImplicitOrEqual(Index index, Index expected) {
  return index == kImplicit || index == expected;
}

/// Divides the lower and upper bounds of `interval` by `divisor`, rounding out
/// (expanding the interval) to the nearest integer.
///
/// \dchecks `divisor > 0`
constexpr inline IndexInterval DividePositiveRoundOut(IndexInterval interval,
                                                      Index divisor) {
  assert(divisor > 0);
  return IndexInterval::UncheckedHalfOpen(
      FloorOfRatio(interval.inclusive_min(), divisor),
      CeilOfRatio(interval.exclusive_max(), divisor));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_INTERVAL_H_
