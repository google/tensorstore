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
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, const OptionalByteRangeRequest& r) {
  os << "[" << r.inclusive_min << ", ";
  if (r.exclusive_max) {
    os << *r.exclusive_max;
  } else {
    os << "?";
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ByteRange& r) {
  return os << "[" << r.inclusive_min << ", " << r.exclusive_max << ")";
}

Result<ByteRange> OptionalByteRangeRequest::Validate(std::uint64_t size) const {
  assert(SatisfiesInvariants());
  if (exclusive_max && *exclusive_max > size) {
    return absl::OutOfRangeError(StrCat("Requested byte range ", *this,
                                        " is not valid for value of size ",
                                        size));
  }
  return ByteRange{inclusive_min, exclusive_max.value_or(size)};
}

}  // namespace tensorstore
