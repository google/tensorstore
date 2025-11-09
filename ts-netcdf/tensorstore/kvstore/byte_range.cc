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

#include "tensorstore/kvstore/byte_range.h"

#include <cassert>
#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_optional.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, const OptionalByteRangeRequest& r) {
  os << "[" << r.inclusive_min << ", ";
  if (r.exclusive_max != -1) {
    os << r.exclusive_max;
  } else {
    os << "?";
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ByteRange& r) {
  return os << "[" << r.inclusive_min << ", " << r.exclusive_max << ")";
}

Result<ByteRange> OptionalByteRangeRequest::Validate(int64_t size) const {
  assert(SatisfiesInvariants());
  int64_t inclusive_min = this->inclusive_min;
  int64_t exclusive_max = this->exclusive_max;
  if (exclusive_max == -1) exclusive_max = size;
  if (inclusive_min < 0) {
    inclusive_min += size;
  }
  if (inclusive_min < 0 || exclusive_max > size ||
      inclusive_min > exclusive_max) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Requested byte range ", *this,
                            " is not valid for value of size ", size));
  }
  return ByteRange{inclusive_min, exclusive_max};
}

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::ByteRange, tensorstore::serialization::ApplyMembersSerializer<
                                tensorstore::ByteRange>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::OptionalByteRangeRequest,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::OptionalByteRangeRequest>())
