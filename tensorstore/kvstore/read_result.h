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

#ifndef TENSORSTORE_KVSTORE_READ_RESULT_H_
#define TENSORSTORE_KVSTORE_READ_RESULT_H_

#include <iosfwd>
#include <optional>

#include "absl/strings/cord.h"
#include "tensorstore/kvstore/generation.h"

namespace tensorstore {
namespace kvstore {

using Key = std::string;
using Value = absl::Cord;

struct ReadResult {
  enum class State {
    /// Indicates an unspecified value, used when a conditional read was
    /// requested and the condition was not satisfied.  The `value` member
    /// must be empty.
    kUnspecified,
    /// Indicates a missing value (not an error).  The `value` member must be
    /// empty.
    kMissing,
    /// Indicates a value is present.
    kValue
  };

  constexpr static State kUnspecified = State::kUnspecified;
  constexpr static State kMissing = State::kMissing;
  constexpr static State kValue = State::kValue;

  friend std::ostream& operator<<(std::ostream& os, State state);

  ReadResult() = default;

  /// Constructs a `ReadResult` with the value unspecified.
  ReadResult(TimestampedStorageGeneration stamp) : stamp(std::move(stamp)) {}

  ReadResult(State state, Value value, TimestampedStorageGeneration stamp)
      : state(state), value(std::move(value)), stamp(std::move(stamp)) {}

  /// Indicates the interpretation of `value`.
  State state = kUnspecified;

  /// Specifies the value if `state == kValue`.  Otherwise must be empty.
  Value value;

  /// Generation and timestamp associated with `value` and `state`.
  ///
  /// The `time` must be greater than or equal to the `staleness_bound`
  /// specified in the `ReadOptions` (or the time of the read request, if a
  /// `staleness_bound` in the future was specified).
  TimestampedStorageGeneration stamp;

  /// Returns `true` if the read was aborted because the conditions were not
  /// satisfied.
  bool aborted() const { return state == kUnspecified; }

  /// Returns `true` if the key was not found.
  bool not_found() const { return state == kMissing; }

  bool has_value() const { return state == kValue; }

  std::optional<Value> optional_value() const& {
    if (state == kValue) return value;
    return std::nullopt;
  }

  std::optional<Value> optional_value() && {
    if (state == kValue) return std::move(value);
    return std::nullopt;
  }

  friend bool operator==(const ReadResult& a, const ReadResult& b) {
    return a.state == b.state && a.value == b.value && a.stamp == b.stamp;
  }
  friend bool operator!=(const ReadResult& a, const ReadResult& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const ReadResult& x);

  // Reflection support.
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.state, x.value, x.stamp);
  };
};

}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_READ_RESULT_H_
