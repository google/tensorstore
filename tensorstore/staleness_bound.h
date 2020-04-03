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

namespace tensorstore {

class StalenessBound : public absl::Time {
 public:
  /// The default StalenessBound of `absl::InfiniteFuture()` ensures data is
  /// never stale.
  StalenessBound() : absl::Time(absl::InfiniteFuture()) {}

  StalenessBound(absl::Time newer_than_time) : absl::Time(newer_than_time) {}

  StalenessBound(absl::Duration duration_before_now)
      : absl::Time(absl::Now() - duration_before_now) {}

  /// Overload for the benefit of GoogleTest.
  friend std::ostream& operator<<(std::ostream& os,
                                  const StalenessBound& bound) {
    return os << static_cast<const absl::Time&>(bound);
  }

  /// Returns the effective bound given the specified open time.
  StalenessBound BoundAtOpen(absl::Time open_time) const {
    StalenessBound result = *this;
    if (result.bounded_by_open_time) {
      static_cast<absl::Time&>(result) = open_time;
    }
    return result;
  }

  /// Returns a new staleness bound with `bounded_at_open_time == true`.
  static StalenessBound BoundedByOpen() {
    StalenessBound b;
    b.bounded_by_open_time = true;
    return b;
  }

  /// Specifies whether this `StalenessBound` was initialized with an "open
  /// time".  When serializing back to JSON, such a bound is recorded specially
  /// rather than as a timestamp.
  bool bounded_by_open_time = false;
};

/// Combines staleness bound for metadata and data.
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
};

}  // namespace tensorstore

#endif  // TENSORSTORE_STALENESS_BOUND_H_
