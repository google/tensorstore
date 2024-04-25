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

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <ostream>

#include "absl/strings/cord.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies a range of bytes within a stored value.
///
/// \ingroup kvstore
struct ByteRange {
  /// Specifies the starting byte (inclusive).
  int64_t inclusive_min;

  /// Specifies the ending byte (exclusive).
  int64_t exclusive_max;

  /// Checks that this byte range is valid.
  constexpr bool SatisfiesInvariants() const {
    return inclusive_min >= 0 && exclusive_max >= 0 &&
           exclusive_max >= inclusive_min;
  }

  /// Returns the number of bytes contained in the range.
  ///
  /// \dchecks `SatisfiesInvariants()`
  int64_t size() const {
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
  /// Constructs a request for a full (unconstrained) byte range.
  ///
  /// \id full
  constexpr OptionalByteRangeRequest() : inclusive_min(0), exclusive_max(-1) {}

  /// Constructs from the specified bounds.
  ///
  /// If `inclusive_min < 0`, it indicates a byte offset relative to the end,
  /// and `exclusive_max` must be left as `-1`.
  ///
  /// If `exclusive_max == -1`, it indicates that the byte range continues until
  /// the end of the value.
  ///
  /// \id inclusive_min, exclusive_max
  explicit constexpr OptionalByteRangeRequest(int64_t inclusive_min,
                                              int64_t exclusive_max = -1)
      : inclusive_min(inclusive_min), exclusive_max(exclusive_max) {}

  /// Constructs from an existing byte range.
  ///
  /// \id ByteRange
  constexpr OptionalByteRangeRequest(ByteRange r)
      : inclusive_min(r.inclusive_min), exclusive_max(r.exclusive_max) {}

  /// Checks if no byte range restriction is specified.
  bool IsFull() const { return inclusive_min == 0 && exclusive_max == -1; }

  /// Checks if this request specifies an explicit range with both
  /// `inclusive_min` and `exclusive_max`.
  bool IsRange() const { return exclusive_max != -1; }

  /// Check if this request specifies a suffix length.
  bool IsSuffixLength() const { return inclusive_min < 0; }

  /// Checks if this request specifies a suffix, with only `inclusive_min`
  /// specified.
  bool IsSuffix() const { return exclusive_max == -1 && inclusive_min > 0; }

  /// Constructs a request for an explicit range.
  static OptionalByteRangeRequest Range(int64_t inclusive_min,
                                        int64_t exclusive_max) {
    assert(inclusive_min >= 0);
    assert(exclusive_max >= 0);
    return OptionalByteRangeRequest{inclusive_min, exclusive_max};
  }

  /// Constructs a request for a suffix of the specified length.
  static OptionalByteRangeRequest SuffixLength(int64_t length) {
    assert(length >= 0);
    return OptionalByteRangeRequest{-length, -1};
  }

  /// Constructs a request for a suffix starting at the specified
  /// `inclusive_min`.
  static OptionalByteRangeRequest Suffix(int64_t inclusive_min) {
    assert(inclusive_min >= 0);
    return OptionalByteRangeRequest{inclusive_min, -1};
  }

  /// Specifies the starting byte if non-negative, or suffix length if negative.
  int64_t inclusive_min = 0;

  /// Specifies the exclusive max, or `-1` to indicate no upper bound.
  int64_t exclusive_max = -1;

  /// Returns the number of bytes contained in the range, or `-1` if
  /// unknown.
  ///
  /// \dchecks `SatisfiesInvariants()`
  int64_t size() const {
    assert(SatisfiesInvariants());
    if (inclusive_min < 0) return -inclusive_min;
    if (exclusive_max != -1) return exclusive_max - inclusive_min;
    return -1;
  }

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

  /// Converts to a `ByteRange`.
  ///
  /// \pre `IsRange()`
  ByteRange AsByteRange() const {
    assert(IsRange());
    return {inclusive_min, exclusive_max};
  }

  /// Prints a debugging string representation to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const OptionalByteRangeRequest& r);

  /// Checks that this byte range is valid.
  constexpr bool SatisfiesInvariants() const {
    return (exclusive_max == -1 ||
            (exclusive_max >= inclusive_min && inclusive_min >= 0));
  }

  /// Returns a `ByteRange` for an object of size.
  ///
  /// \error `absl::StatusCode::kOutOfRange` if `inclusive_min` or
  ///   `*exclusive_max` are not within the object size.
  Result<ByteRange> Validate(int64_t size) const;

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
