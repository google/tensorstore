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

#include "tensorstore/index_interval.h"

#include <ostream>

#include "absl/status/status.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

Result<IndexInterval> IndexInterval::Closed(Index inclusive_min,
                                            Index inclusive_max) {
  if (!ValidClosed(inclusive_min, inclusive_max)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("(", inclusive_min, ", ", inclusive_max,
                            ") do not specify a valid closed index interval"));
  }
  return UncheckedClosed(inclusive_min, inclusive_max);
}

Result<IndexInterval> IndexInterval::HalfOpen(Index inclusive_min,
                                              Index exclusive_max) {
  if (!ValidHalfOpen(inclusive_min, exclusive_max)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "(", inclusive_min, ", ", exclusive_max,
        ") do not specify a valid half-open index interval"));
  }
  return UncheckedHalfOpen(inclusive_min, exclusive_max);
}

Result<IndexInterval> IndexInterval::Sized(Index inclusive_min, Index size) {
  if (!ValidSized(inclusive_min, size)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("(", inclusive_min, ", ", size,
                            ") do not specify a valid sized index interval"));
  }
  return UncheckedSized(inclusive_min, size);
}

std::ostream& operator<<(std::ostream& os,
                         const OptionallyImplicitIndexInterval& x) {
  if (x.inclusive_min() == -kInfIndex) {
    os << "(-inf";
  } else {
    os << '[' << x.inclusive_min();
  }
  if (x.implicit_lower()) os << '*';
  os << ", ";
  if (x.inclusive_max() == +kInfIndex) {
    os << "+inf";
  } else {
    os << x.exclusive_max();
  }
  if (x.implicit_upper()) os << '*';
  return os << ")";
}

std::ostream& operator<<(std::ostream& os, IndexInterval x) {
  return os << OptionallyImplicitIndexInterval(x, false, false);
}

namespace {
template <ContainerKind CKindA, ContainerKind CKindB>
bool EqualImpl(const IndexDomainDimension<CKindA>& a,
               const IndexDomainDimension<CKindB>& b) {
  return (a.optionally_implicit_interval() ==
              b.optionally_implicit_interval() &&
          a.label() == b.label());
}
}  // namespace

bool operator==(const IndexDomainDimension<container>& a,
                const IndexDomainDimension<container>& b) {
  return EqualImpl(a, b);
}

bool operator==(const IndexDomainDimension<view>& a,
                const IndexDomainDimension<view>& b) {
  return EqualImpl(a, b);
}

bool operator==(const IndexDomainDimension<view>& a,
                const IndexDomainDimension<container>& b) {
  return EqualImpl(a, b);
}

bool operator==(const IndexDomainDimension<container>& a,
                const IndexDomainDimension<view>& b) {
  return EqualImpl(a, b);
}

std::ostream& operator<<(std::ostream& os,
                         const IndexDomainDimension<view>& x) {
  if (!x.label().empty()) {
    os << QuoteString(x.label()) << ": ";
  }
  return os << x.optionally_implicit_interval();
}

std::ostream& operator<<(std::ostream& os,
                         const IndexDomainDimension<container>& x) {
  return os << IndexDomainDimension<view>(x);
}

bool AreCompatibleOrUnbounded(IndexInterval a, IndexInterval b) {
  Index a_lower = a.inclusive_min();
  Index a_upper = a.inclusive_max();
  Index b_lower = b.inclusive_min();
  Index b_upper = b.inclusive_max();
  return (a_lower == b_lower || a_lower == -kInfIndex ||
          b_lower == -kInfIndex) &&
         (a_upper == b_upper || a_upper == kInfIndex || b_upper == kInfIndex);
}

IndexInterval Hull(IndexInterval a, IndexInterval b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  const Index lower = std::min(a.inclusive_min(), b.inclusive_min());
  const Index upper = std::max(a.inclusive_max(), b.inclusive_max());
  return IndexInterval::UncheckedClosed(lower, upper);
}

IndexInterval Intersect(IndexInterval a, IndexInterval b) {
  const Index lower = std::max(a.inclusive_min(), b.inclusive_min());
  const Index upper = std::min(a.inclusive_max(), b.inclusive_max());
  const Index size = upper < lower ? 0 : upper - lower + 1;
  return IndexInterval::UncheckedSized(lower, size);
}

OptionallyImplicitIndexInterval Hull(OptionallyImplicitIndexInterval a,
                                     OptionallyImplicitIndexInterval b) {
  IndexInterval interval = Hull(a.interval(), b.interval());
  bool implicit_lower = (a.inclusive_min() == b.inclusive_min())
                            ? (a.implicit_lower() && b.implicit_lower())
                            : (interval.inclusive_min() == a.inclusive_min()
                                   ? a.implicit_lower()
                                   : b.implicit_lower());
  bool implicit_upper = (a.inclusive_max() == b.inclusive_max())
                            ? (a.implicit_upper() && b.implicit_upper())
                            : (a.inclusive_max() == interval.inclusive_max()
                                   ? a.implicit_upper()
                                   : b.implicit_upper());

  return OptionallyImplicitIndexInterval{interval, implicit_lower,
                                         implicit_upper};
}

OptionallyImplicitIndexInterval Intersect(OptionallyImplicitIndexInterval a,
                                          OptionallyImplicitIndexInterval b) {
  IndexInterval interval = Intersect(a.interval(), b.interval());
  bool implicit_lower = (a.inclusive_min() == b.inclusive_min())
                            ? (a.implicit_lower() && b.implicit_lower())
                            : (interval.inclusive_min() == a.inclusive_min()
                                   ? a.implicit_lower()
                                   : b.implicit_lower());
  bool implicit_upper = (a.inclusive_max() == b.inclusive_max())
                            ? (a.implicit_upper() && b.implicit_upper())
                            : (a.inclusive_max() == interval.inclusive_max()
                                   ? a.implicit_upper()
                                   : b.implicit_upper());

  return OptionallyImplicitIndexInterval{interval, implicit_lower,
                                         implicit_upper};
}

OptionallyImplicitIndexInterval IntersectPreferringExplicit(
    OptionallyImplicitIndexInterval a, OptionallyImplicitIndexInterval b) {
  const Index inclusive_min =
      a.implicit_lower() == b.implicit_lower()
          ? std::max(a.inclusive_min(), b.inclusive_min())
          : std::max(a.effective_interval().inclusive_min(),
                     b.effective_interval().inclusive_min());
  const Index inclusive_max =
      a.implicit_upper() == b.implicit_upper()
          ? std::min(a.inclusive_max(), b.inclusive_max())
          : std::min(a.effective_interval().inclusive_max(),
                     b.effective_interval().inclusive_max());
  return OptionallyImplicitIndexInterval{
      IndexInterval::UncheckedClosed(
          inclusive_min, std::max(inclusive_min - 1, inclusive_max)),
      a.implicit_lower() && b.implicit_lower(),
      a.implicit_upper() && b.implicit_upper()};
}

bool ContainsOrUnbounded(IndexInterval outer, IndexInterval inner) {
  return (inner.inclusive_min() == -kInfIndex ||
          inner.inclusive_min() >= outer.inclusive_min()) &&
         (inner.inclusive_max() == kInfIndex ||
          inner.inclusive_max() <= outer.inclusive_max());
}

Result<IndexInterval> ShiftInterval(IndexInterval interval, Index min_offset,
                                    Index max_offset) {
  Index inclusive_min;
  if (interval.inclusive_min() == -kInfIndex) {
    inclusive_min = -kInfIndex;
  } else if (internal::AddOverflow(interval.inclusive_min(), min_offset,
                                   &inclusive_min) ||
             !IsFiniteIndex(inclusive_min)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        interval.inclusive_min(), " + ", min_offset, " is outside valid range ",
        IndexInterval::FiniteRange()));
  }
  Index inclusive_max;
  if (interval.inclusive_max() == kInfIndex) {
    inclusive_max = kInfIndex;
  } else if (internal::AddOverflow(interval.inclusive_max(), max_offset,
                                   &inclusive_max) ||
             !IsFiniteIndex(inclusive_max)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        interval.inclusive_max(), " + ", max_offset, " is outside valid range ",
        IndexInterval::FiniteRange()));
  }
  return IndexInterval::UncheckedClosed(inclusive_min, inclusive_max);
}

Result<IndexInterval> ShiftIntervalBackward(IndexInterval interval,
                                            Index min_offset,
                                            Index max_offset) {
  // If `min_offset` or `max_offset` equals `std::numeric_limits<Index>::min()`,
  // then wrapping negation is a no-op (normal `operator-` in that case is
  // undefined behavior due to signed integer overflow).  However, it turns out
  // `ShiftInterval` still gives the correct result in the case that the offset
  // is incorrectly wrapped, because the offset is either ignored due to the
  // corresponding bound being infinite, or it leads to overflow.  In the case
  // of overflow, the error message will include the incorrectly-wrapped value,
  // but that is not terribly important.
  return ShiftInterval(
      interval, internal::wrap_on_overflow::Multiply(min_offset, Index(-1)),
      internal::wrap_on_overflow::Multiply(max_offset, Index(-1)));
}

Result<IndexInterval> ShiftInterval(IndexInterval interval, Index offset) {
  return ShiftInterval(interval, offset, offset);
}

Result<IndexInterval> ShiftIntervalBackward(IndexInterval interval,
                                            Index offset) {
  return ShiftIntervalBackward(interval, offset, offset);
}

Result<IndexInterval> ShiftIntervalTo(IndexInterval interval, Index origin) {
  if (!IsFiniteIndex(origin)) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Origin ", origin, " is outside valid range ",
                            IndexInterval::FiniteRange()));
  }
  if (interval.inclusive_min() == -kInfIndex) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Interval ", interval, " is not bounded below"));
  }
  // Guaranteed not to overflow because `IsFiniteIndex(origin) == true`.
  Index offset;
  [[maybe_unused]] const bool overflow =
      internal::SubOverflow(origin, interval.inclusive_min(), &offset);
  assert(!overflow);
  return ShiftInterval(interval, offset);
}

absl::Status CheckContains(IndexInterval interval, Index index) {
  if (Contains(interval, index)) return absl::OkStatus();
  return absl::OutOfRangeError(tensorstore::StrCat(
      "Index ", index, " is outside valid range ", interval));
}

Result<std::pair<OptionallyImplicitIndexInterval, Index>> ExtractStridedSlice(
    OptionallyImplicitIndexInterval orig, IntervalForm interval_form,
    Index start, Index stop_or_size, Index stride) {
  // We define `constraint` here because `orig` is modified later.
  const IndexInterval constraint = IndexInterval::UncheckedClosed(
      orig.implicit_lower() ? -kInfIndex : orig.inclusive_min(),
      orig.implicit_upper() ? kInfIndex : orig.inclusive_max());

  // Check for 0 and std::numeric_limits<Index>::min(), which are both invalid.
  if (stride == 0 || stride == std::numeric_limits<Index>::min()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid stride ", stride));
  }
  if (start == kImplicit) {
    start = stride > 0 ? orig.inclusive_min() : orig.inclusive_max();
  } else {
    if (!IsValidIndex(start)) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Invalid start index ", start));
    }
    orig.implicit_lower() = false;
  }
  Index inclusive_stop;
  if (interval_form == IntervalForm::sized) {
    Index size = stop_or_size;
    if (size == kImplicit) {
      inclusive_stop = stride > 0 ? orig.inclusive_max() : orig.inclusive_min();
    } else {
      if (size < 0) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Negative size ", size, " specified for sized interval"));
      }
      orig.implicit_upper() = false;
      if (size == 0) {
        // Can't overflow since `IsValidIndex(start) == true`.
        inclusive_stop = start + (stride > 0 ? -1 : 1);
      } else {
        // One less than the size of the slice in the original input domain.
        if (internal::MulOverflow(stride, size - 1, &inclusive_stop) ||
            internal::AddOverflow(start, inclusive_stop, &inclusive_stop)) {
          return absl::OutOfRangeError(
              tensorstore::StrCat("Integer overflow computing slice result"));
        }
      }
    }
  } else {
    if (stop_or_size == kImplicit) {
      inclusive_stop = stride > 0 ? orig.inclusive_max() : orig.inclusive_min();
    } else {
      orig.implicit_upper() = false;
      if (interval_form == IntervalForm::closed ||
          !IsFiniteIndex(stop_or_size)) {
        inclusive_stop = stop_or_size;
      } else {
        assert(interval_form == IntervalForm::half_open);
        // Can't overflow since `IsFiniteIndex(stop_or_size) == true`.
        inclusive_stop = stop_or_size + (stride > 0 ? -1 : 1);
      }
    }
  }
  if (std::abs(stride) != 1 && !IsFiniteIndex(start)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Slicing with non-unit stride of ", stride,
                            " requires a finite start index"));
  }
  Index adjusted_inclusive_min, adjusted_inclusive_max;
  if (stride > 0) {
    adjusted_inclusive_min = start;
    adjusted_inclusive_max = inclusive_stop;
  } else {
    adjusted_inclusive_min = inclusive_stop;
    adjusted_inclusive_max = start;
    std::swap(orig.implicit_lower(), orig.implicit_upper());
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto adjusted_interval,
      IndexInterval::Closed(adjusted_inclusive_min, adjusted_inclusive_max));
  if (!Contains(constraint, adjusted_interval)) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Slice interval ", adjusted_interval,
                            " is not contained within domain ", constraint));
  }

  Index new_start = start / stride;
  Index new_size =
      std::abs(inclusive_stop) == kInfIndex
          ? kInfIndex + 1 - new_start
          : CeilOfRatio(adjusted_interval.size(), std::abs(stride));
  orig.interval() = IndexInterval::UncheckedSized(new_start, new_size);
  return {std::in_place, orig, start};
}

Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractHalfOpenStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                            Index stop, Index stride) {
  return ExtractStridedSlice(orig, IntervalForm::half_open, start, stop,
                             stride);
}

Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractClosedStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                          Index stop, Index stride) {
  return ExtractStridedSlice(orig, IntervalForm::closed, start, stop, stride);
}

Result<std::pair<OptionallyImplicitIndexInterval, Index>>
ExtractSizedStridedSlice(OptionallyImplicitIndexInterval orig, Index start,
                         Index size, Index stride) {
  return ExtractStridedSlice(orig, IntervalForm::sized, start, size, stride);
}

absl::Status ComputeStridedSliceMap(OptionallyImplicitIndexInterval orig,
                                    IntervalForm interval_form,
                                    Index translate_origin_to, Index start,
                                    Index stop_or_size, Index stride,
                                    OptionallyImplicitIndexInterval* new_domain,
                                    Index* output_offset) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_interval_and_adjusted_start,
      ExtractStridedSlice(orig, interval_form, start, stop_or_size, stride));
  OptionallyImplicitIndexInterval& new_interval =
      new_interval_and_adjusted_start.first;
  Index adjusted_start = new_interval_and_adjusted_start.second;

  if (translate_origin_to != kImplicit) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        new_interval.interval(),
        ShiftIntervalTo(new_interval.interval(), translate_origin_to));
  }
  *new_domain = new_interval;
  *output_offset = adjusted_start - new_interval.inclusive_min() * stride;
  return absl::OkStatus();
}

Result<IndexInterval> GetAffineTransformDomain(IndexInterval interval,
                                               Index offset, Index divisor) {
  assert(divisor != 0);
  if (interval == IndexInterval()) {
    // Special case for fully unbounded interval.  Divisor and offset don't need
    // to be checked in this case (and the checks below are too restrictive for
    // this case).
    return interval;
  }
  do {
    Index result_lower, result_size;
    Index lower, upper;

    if (divisor < 0) {
      if (divisor == std::numeric_limits<Index>::min() ||
          offset == std::numeric_limits<Index>::min()) {
        break;
      }
      divisor = -divisor;
      offset = -offset;
      lower = -interval.inclusive_max();
      upper = -interval.inclusive_min();
      if (interval.empty()) {
        --lower;
        --upper;
      }
    } else {
      lower = interval.inclusive_min();
      upper = interval.inclusive_max();
    }

    if (lower == -kInfIndex) {
      result_lower = -kInfIndex;
    } else {
      if (internal::SubOverflow(lower, offset, &result_lower)) break;
      result_lower = CeilOfRatio(result_lower, divisor);
      if (!IsFiniteIndex(result_lower)) break;
    }
    if (interval.empty()) {
      result_size = 0;
    } else if (upper == kInfIndex) {
      result_size = kInfIndex - result_lower + 1;
    } else {
      Index result_upper;
      if (internal::SubOverflow(upper, offset, &result_upper)) break;
      result_upper = FloorOfRatio(result_upper, divisor);
      if (!IsFiniteIndex(result_upper)) break;
      result_size = result_upper - result_lower + 1;
    }
    return IndexInterval::UncheckedSized(result_lower, result_size);
  } while (false);
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Integer overflow propagating range ", interval,
                          " through inverse affine transform with offset ",
                          offset, " and multiplier ", divisor));
}

Result<OptionallyImplicitIndexInterval> GetAffineTransformDomain(
    OptionallyImplicitIndexInterval interval, Index offset, Index divisor) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      interval.interval(),
      GetAffineTransformDomain(interval.interval(), offset, divisor));
  if (divisor < 0) {
    std::swap(interval.implicit_lower(), interval.implicit_upper());
  }
  return interval;
}

namespace {
absl::Status GetAffineTransformError(IndexInterval interval, Index offset,
                                     Index multiplier) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Integer overflow computing affine transform of domain ", interval,
      " with offset ", offset, " and multiplier ", multiplier));
}
}  // namespace

Result<IndexInterval> GetAffineTransformRange(IndexInterval interval,
                                              Index offset, Index multiplier) {
  const auto transform_bound_overflow = [&](Index* bound) {
    if (*bound == -kInfIndex || *bound == kInfIndex) {
      if (multiplier < 0) *bound *= -1;
      return false;
    }
    return (internal::MulOverflow(*bound, multiplier, bound) ||
            internal::AddOverflow(*bound, offset, bound) ||
            !IsFiniteIndex(*bound));
  };

  Index lower = interval.inclusive_min(), upper = interval.inclusive_max();
  if (transform_bound_overflow(&lower) || transform_bound_overflow(&upper)) {
    return GetAffineTransformError(interval, offset, multiplier);
  }
  if (interval.empty()) {
    return IndexInterval::UncheckedSized(lower, 0);
  }
  if (multiplier == 0) {
    return IndexInterval::UncheckedSized(lower, 1);
  }
  if (multiplier < 0) std::swap(lower, upper);
  return IndexInterval::UncheckedClosed(lower, upper);
}

Result<IndexInterval> GetAffineTransformInverseDomain(IndexInterval interval,
                                                      Index offset,
                                                      Index divisor) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_interval, GetAffineTransformRange(interval, offset, divisor));
  if (new_interval.empty()) return new_interval;
  if (divisor > 0 && new_interval.inclusive_max() != kInfIndex) {
    Index new_inclusive_max;
    if (internal::AddOverflow(new_interval.inclusive_max(), divisor - 1,
                              &new_inclusive_max) ||
        !IsFiniteIndex(new_inclusive_max)) {
      return GetAffineTransformError(interval, offset, divisor);
    }
    return IndexInterval::UncheckedClosed(new_interval.inclusive_min(),
                                          new_inclusive_max);
  }
  if (divisor < 0 && new_interval.inclusive_min() != -kInfIndex) {
    Index new_inclusive_min;
    if (internal::AddOverflow(new_interval.inclusive_min(), divisor + 1,
                              &new_inclusive_min) ||
        !IsFiniteIndex(new_inclusive_min)) {
      return GetAffineTransformError(interval, offset, divisor);
    }
    return IndexInterval::UncheckedClosed(new_inclusive_min,
                                          new_interval.inclusive_max());
  }
  return new_interval;
}

Result<OptionallyImplicitIndexInterval> GetAffineTransformRange(
    OptionallyImplicitIndexInterval interval, Index offset, Index multiplier) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      interval.interval(),
      GetAffineTransformRange(interval.interval(), offset, multiplier));
  if (multiplier < 0) {
    std::swap(interval.implicit_lower(), interval.implicit_upper());
  }
  return interval;
}

Result<std::string_view> MergeDimensionLabels(std::string_view a,
                                              std::string_view b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a == b) return a;
  return absl::InvalidArgumentError("Dimension labels do not match");
}

Result<OptionallyImplicitIndexInterval> MergeOptionallyImplicitIndexIntervals(
    OptionallyImplicitIndexInterval a, OptionallyImplicitIndexInterval b) {
  if (a == b) return a;
  Index inclusive_min, inclusive_max;
  bool implicit_lower, implicit_upper;
  if (a.inclusive_min() == -kInfIndex && a.implicit_lower() == true) {
    inclusive_min = b.inclusive_min();
    implicit_lower = b.implicit_lower();
  } else if (b.inclusive_min() == -kInfIndex && b.implicit_lower() == true) {
    inclusive_min = a.inclusive_min();
    implicit_lower = a.implicit_lower();
  } else if (a.inclusive_min() != b.inclusive_min()) {
    return absl::InvalidArgumentError("Lower bounds do not match");
  } else {
    inclusive_min = a.inclusive_min();
    implicit_lower = a.implicit_lower() && b.implicit_lower();
  }
  if (a.inclusive_max() == kInfIndex && a.implicit_upper() == true) {
    inclusive_max = b.inclusive_max();
    implicit_upper = b.implicit_upper();
  } else if (b.inclusive_max() == kInfIndex && b.implicit_upper() == true) {
    inclusive_max = a.inclusive_max();
    implicit_upper = a.implicit_upper();
  } else if (a.inclusive_max() != b.inclusive_max()) {
    return absl::InvalidArgumentError("Upper bounds do not match");
  } else {
    inclusive_max = a.inclusive_max();
    implicit_upper = a.implicit_upper() && b.implicit_upper();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto interval, IndexInterval::Closed(inclusive_min, inclusive_max));
  return OptionallyImplicitIndexInterval{interval, implicit_lower,
                                         implicit_upper};
}

namespace serialization {

bool Serializer<IndexInterval>::Encode(EncodeSink& sink,
                                       const IndexInterval& value) {
  return serialization::EncodeTuple(sink, value.inclusive_min(), value.size());
}

bool Serializer<IndexInterval>::Decode(DecodeSource& source,
                                       IndexInterval& value) {
  Index inclusive_min, size;
  if (!serialization::DecodeTuple(source, inclusive_min, size)) {
    return false;
  }
  TENSORSTORE_ASSIGN_OR_RETURN(value, IndexInterval::Sized(inclusive_min, size),
                               (source.Fail(_), false));
  return true;
}

}  // namespace serialization

}  // namespace tensorstore
