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

#ifndef TENSORSTORE_STALENESS_BOUND_H_
#define TENSORSTORE_STALENESS_BOUND_H_

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {

/// Specifies time bound on cached data that may be used without revalidation.
///
/// \relates Spec
struct RecheckCacheOption {
  /// Constructs an unspecified bound.
  ///
  /// \id default
  constexpr explicit RecheckCacheOption() {}

  /// Constructs a bound that either always or never revalidates cached data.
  ///
  /// \param value If `true`, always revalidate cached data regardless of how
  ///     old it is.  If `false`, never revalidate cached data, regardless of
  ///     how old it is.
  /// \id bool
  constexpr explicit RecheckCacheOption(bool value)
      : RecheckCacheOption(value ? absl::InfiniteFuture()
                                 : absl::InfinitePast()) {}

  /// Constructs from the specified bound.
  ///
  /// \id time
  constexpr explicit RecheckCacheOption(absl::Time time)
      : time(time), flags(kSpecified) {}

  /// Special time bound equal to the time the TensorStore is opened.
  static constexpr RecheckCacheOption AtOpen() {
    RecheckCacheOption option;
    option.flags = kAtOpen;
    return option;
  }

  /// Specifies the kind of time bound.
  enum Flags {
    /// No bound has been specified.
    kUnspecified,
    /// Data must not be older than the specified `time`.
    kSpecified,
    /// Data must not be older than the time at which the Spec is opened.
    kAtOpen,
  };

  /// Specifies the time bound.
  ///
  /// If `flags == kSpecified`, data must not be older than `time`.
  absl::Time time = absl::InfiniteFuture();

  /// Specifies the interpretation of `time`.
  Flags flags = kUnspecified;

  /// Checks if a bound has been specified.
  constexpr bool specified() const { return flags != kUnspecified; }
};

/// Specifies time bound on cached array data (as opposed to metadata).
///
/// This is for use with `SpecOptions` and is applicable to drivers that perform
/// caching.
///
/// `RecheckCachedData{time}` indicates that cached data older than `time` will
/// be rechecked before it is returned; data not older than `time` is assumed to
/// be valid (no requests are made to validate it).
///
/// `RecheckCachedData{true}` indicates that cached data is always rechecked
/// before it is returned, and is equivalent to
/// `RecheckCachedData{absl::InfiniteFuture()}`.
///
/// `RecheckCachedData{false}` indicates that cached data is always returned
/// without rechecking, and is equivalent to
/// `RecheckCachedData{absl::InfiniteFuture()}`.
///
/// `RecheckCachedData::AtOpen()` indicates that cached older than the time at
/// which the TensorStore was opened will be rechecked before it is returned;
/// data not older than the open time is assumed to be valid.
///
/// \relates Spec
struct RecheckCachedData : public RecheckCacheOption {
  constexpr explicit RecheckCachedData(RecheckCacheOption option)
      : RecheckCacheOption(option) {}
  using RecheckCacheOption::RecheckCacheOption;

  /// Special time bound equal to the time the TensorStore is opened.
  static constexpr RecheckCachedData AtOpen() {
    return RecheckCachedData(RecheckCacheOption::AtOpen());
  }
};

/// Specifies time bound on cached metadata (as opposed to actual array data).
///
/// This is for use with `SpecOptions` and is applicable to drivers that perform
/// caching, and distinguish between data and metadata.
///
/// The usage is the same as for `RecheckCachedData`.
///
/// \relates Spec
struct RecheckCachedMetadata : public RecheckCacheOption {
  constexpr explicit RecheckCachedMetadata(RecheckCacheOption option)
      : RecheckCacheOption(option) {}
  using RecheckCacheOption::RecheckCacheOption;
  static constexpr RecheckCachedMetadata AtOpen() {
    return RecheckCachedMetadata(RecheckCacheOption::AtOpen());
  }
};

/// Specifies the same time bound for both cached array data and metadata.
///
/// This is for use with `SpecOptions` and is applicable to drivers that perform
/// caching.
///
/// The usage is the same as for `RecheckCachedData`.
///
/// \relates Spec
struct RecheckCached : public RecheckCacheOption {
  constexpr explicit RecheckCached(RecheckCacheOption option)
      : RecheckCacheOption(option) {}
  using RecheckCacheOption::RecheckCacheOption;

  /// Special time bound equal to the time the TensorStore is opened.
  static constexpr RecheckCached AtOpen() {
    return RecheckCached(RecheckCacheOption::AtOpen());
  }
};

class StalenessBound {
 public:
  /// The default StalenessBound of `absl::InfiniteFuture()` ensures data is
  /// never stale.
  StalenessBound() = default;

  StalenessBound(RecheckCacheOption option)
      : time(option.time),
        bounded_by_open_time(option.flags == RecheckCacheOption::kAtOpen) {}

  StalenessBound(absl::Time newer_than_time) : time(newer_than_time) {}

  StalenessBound(absl::Duration duration_before_now)
      : time(absl::Now() - duration_before_now) {}

  /// Overload for the benefit of GoogleTest.
  friend std::ostream& operator<<(std::ostream& os,
                                  const StalenessBound& bound) {
    return os << bound.time;
  }

  /// Returns the effective bound given the specified open time.
  StalenessBound BoundAtOpen(absl::Time open_time) const {
    StalenessBound result = *this;
    if (result.bounded_by_open_time) {
      result.time = open_time;
    }
    return result;
  }

  /// Returns a new staleness bound with `bounded_at_open_time == true`.
  static StalenessBound BoundedByOpen() {
    StalenessBound b;
    b.bounded_by_open_time = true;
    return b;
  }

  /// Time bound.
  absl::Time time = absl::InfiniteFuture();

  /// Specifies whether this `StalenessBound` was initialized with an "open
  /// time".  When serializing back to JSON, such a bound is recorded specially
  /// rather than as a timestamp.
  bool bounded_by_open_time = false;

  friend bool operator==(const StalenessBound& a, const StalenessBound& b) {
    return a.time == b.time && a.bounded_by_open_time == b.bounded_by_open_time;
  }

  friend bool operator!=(const StalenessBound& a, const StalenessBound& b) {
    return !(a == b);
  }

  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.time, x.bounded_by_open_time);
  };
};

// Combines staleness bound for metadata and data.
struct StalenessBounds {
  /// Initializes the `metadata` and `data` bound to `absl::InfiniteFuture()`,
  /// which ensures stale data is never seen.
  StalenessBounds() = default;

  /// Initializes the `metadata` and `data` bounds separately.
  StalenessBounds(StalenessBound metadata, StalenessBound data)
      : metadata(metadata), data(data) {}

  /// Initializes both the `metadata` and `data` bounds to `bound`.
  StalenessBounds(StalenessBound bound) : metadata(bound), data(bound) {}

  /// Initializes both the `metadata` and `data` bounds to `newer_than_time`.
  StalenessBounds(absl::Time newer_than_time)
      : StalenessBounds(StalenessBound(newer_than_time)) {}

  /// Initializes both the `metadata` and `data` bounds to
  /// `duration_before_now`.
  StalenessBounds(absl::Duration duration_before_now)
      : StalenessBounds(StalenessBound(duration_before_now)) {}

  StalenessBound metadata;
  StalenessBound data;

  StalenessBounds BoundAtOpen(absl::Time open_time) const {
    return StalenessBounds(metadata.BoundAtOpen(open_time),
                           data.BoundAtOpen(open_time));
  }

  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.metadata, x.data);
  };
};

}  // namespace tensorstore

#endif  // TENSORSTORE_STALENESS_BOUND_H_
