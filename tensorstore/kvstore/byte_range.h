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

#ifndef TENSORSTORE_KVSTORE_BYTE_RANGE_REQUEST_H_
#define TENSORSTORE_KVSTORE_BYTE_RANGE_REQUEST_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/cord.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

struct ByteRange {
  /// Specifies the starting byte.
  std::uint64_t inclusive_min;

  /// Specifies the ending byte.
  std::uint64_t exclusive_max;

  constexpr bool SatisfiesInvariants() const {
    return exclusive_max >= inclusive_min;
  }

  std::uint64_t size() const {
    assert(SatisfiesInvariants());
    return exclusive_max - inclusive_min;
  }

  friend bool operator==(const ByteRange& a, const ByteRange& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }

  friend bool operator!=(const ByteRange& a, const ByteRange& b) {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os, const ByteRange& r);
};

/// Specifies an optional byte range request.
struct OptionalByteRangeRequest {
  OptionalByteRangeRequest(
      std::uint64_t inclusive_min = 0,
      std::optional<std::uint64_t> exclusive_max = std::nullopt)
      : inclusive_min(inclusive_min), exclusive_max(exclusive_max) {}

  OptionalByteRangeRequest(ByteRange r)
      : inclusive_min(r.inclusive_min), exclusive_max(r.exclusive_max) {}

  /// Specifies the starting byte.
  std::uint64_t inclusive_min = 0;

  /// Specifies an optional exclusive ending byte.  If not specified, the full
  /// byte range starting at `inclusive_min` is retrieved.
  /// \invariant `exclusive_max >= inclusive_min`
  std::optional<std::uint64_t> exclusive_max;

  friend bool operator==(const OptionalByteRangeRequest& a,
                         const OptionalByteRangeRequest& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }

  friend bool operator!=(const OptionalByteRangeRequest& a,
                         const OptionalByteRangeRequest& b) {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const OptionalByteRangeRequest& r);

  constexpr bool SatisfiesInvariants() const {
    return (!exclusive_max || exclusive_max >= inclusive_min);
  }

  /// Validates that `*this` is a valid byte range for a value of the specified
  /// `size`.
  ///
  /// \error `absl::StatusCode::kOutOfRange` if `*this` is not valid.
  Result<ByteRange> Validate(std::uint64_t size) const;
};

namespace internal {

/// Returns a sub-cord for a given byte range.
inline absl::Cord GetSubCord(const absl::Cord& s, ByteRange r) {
  assert(r.SatisfiesInvariants());
  const size_t size = s.size();
  assert(r.exclusive_max <= size);
  if (r.inclusive_min == 0 && r.size() == size) return s;
  return s.Subcord(r.inclusive_min, r.size());
}

}  // namespace internal

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_BYTE_RANGE_REQUEST_H_
