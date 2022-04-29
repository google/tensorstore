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
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Specifies a range of bytes within a stored value.
///
/// \ingroup kvstore
struct ByteRange {
  /// Specifies the starting byte (inclusive).
  std::uint64_t inclusive_min;

  /// Specifies the ending byte (exclusive).
  std::uint64_t exclusive_max;

  /// Checks that this byte range is valid.
  constexpr bool SatisfiesInvariants() const {
    return exclusive_max >= inclusive_min;
  }

  /// Returns the number of bytes contained in the range.
  ///
  /// \dchecks `SatisfiesInvariants()`
  std::uint64_t size() const {
    assert(SatisfiesInvariants());
    return exclusive_max - inclusive_min;
  }

  /// Compares two byte ranges for equality.
  friend bool operator==(const ByteRange& a, const ByteRange& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }
  friend bool operator!=(const ByteRange& a, const ByteRange& b) {
    return !(a == b);
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const ByteRange& r);

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.inclusive_min, x.exclusive_max);
  };
};

/// Specifies an optional byte range request.
///
/// \ingroup kvstore
struct OptionalByteRangeRequest {
  /// Constructs from the specified bounds.
  ///
  /// \id inclusive_min, exclusive_max
  OptionalByteRangeRequest(
      std::uint64_t inclusive_min = 0,
      std::optional<std::uint64_t> exclusive_max = std::nullopt)
      : inclusive_min(inclusive_min), exclusive_max(exclusive_max) {}

  /// Constructs from an existing byte range.
  ///
  /// \id ByteRange
  OptionalByteRangeRequest(ByteRange r)
      : inclusive_min(r.inclusive_min), exclusive_max(r.exclusive_max) {}

  /// Specifies the starting byte.
  std::uint64_t inclusive_min = 0;

  /// Specifies an optional exclusive ending byte.  If not specified, the full
  /// byte range starting at `inclusive_min` is retrieved.
  /// \invariant `exclusive_max >= inclusive_min`
  std::optional<std::uint64_t> exclusive_max;

  /// Compares for equality.
  friend bool operator==(const OptionalByteRangeRequest& a,
                         const OptionalByteRangeRequest& b) {
    return a.inclusive_min == b.inclusive_min &&
           a.exclusive_max == b.exclusive_max;
  }
  friend bool operator!=(const OptionalByteRangeRequest& a,
                         const OptionalByteRangeRequest& b) {
    return !(a == b);
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const OptionalByteRangeRequest& r);

  /// Checks that this byte range is valid.
  constexpr bool SatisfiesInvariants() const {
    return (!exclusive_max || exclusive_max >= inclusive_min);
  }

  /// Validates that `*this` is a valid byte range for a value of the specified
  /// `size`.
  ///
  /// \error `absl::StatusCode::kOutOfRange` if `*this` is not valid.
  Result<ByteRange> Validate(std::uint64_t size) const;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.inclusive_min, x.exclusive_max);
  };
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

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::ByteRange)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::OptionalByteRangeRequest)

#endif  // TENSORSTORE_KVSTORE_BYTE_RANGE_REQUEST_H_
