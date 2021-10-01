// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/serialization/absl_time.h"

#include <cstdint>
#include <limits>

#include "absl/time/time.h"
#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace serialization {

// `absl::Duration` is represented as a signed 64-bit integer number of seconds,
// plus an unsigned integer number of quarter nanoseconds (represented as a
// 32-bit unsigned integer).  Additionally, positive and negative infinity are
// represented by `(-std::numeric_limits<int64_t>::min, ~0)` and
// `(-std::numeric_limits<int64_t>::max, ~0)`, respectively.
//
// Since `absl::Duration` does not provide a convenient public API that allows
// for lossless serialization, other than string formatting/parsing, we rely on
// internal APIs in the `absl::time_internal` namespace.
bool Serializer<absl::Duration>::Encode(EncodeSink& sink,
                                        const absl::Duration& value) {
  int64_t rep_hi = absl::time_internal::GetRepHi(value);
  uint32_t rep_lo = absl::time_internal::GetRepLo(value);
  return serialization::EncodeTuple(sink, rep_hi, rep_lo);
}

bool Serializer<absl::Duration>::Decode(DecodeSource& source,
                                        absl::Duration& value) {
  int64_t rep_hi;
  uint32_t rep_lo;
  using absl::time_internal::kTicksPerSecond;
  if (!serialization::DecodeTuple(source, rep_hi, rep_lo)) return false;
  // Verify that representation is valid.
  if (rep_lo >= kTicksPerSecond &&
      (rep_lo != std::numeric_limits<uint32_t>::max() ||
       (rep_hi != std::numeric_limits<int64_t>::min() &&
        rep_hi != std::numeric_limits<int64_t>::max()))) {
    source.Fail(serialization::DecodeError("Invalid time representation"));
    return false;
  }
  value = absl::time_internal::MakeDuration(rep_hi, rep_lo);
  return true;
}

bool Serializer<absl::Time>::Encode(EncodeSink& sink, const absl::Time& value) {
  return serialization::Encode(sink, value - absl::UnixEpoch());
}

bool Serializer<absl::Time>::Decode(DecodeSource& source, absl::Time& value) {
  absl::Duration d;
  if (!serialization::Decode(source, d)) return false;
  value = absl::UnixEpoch() + d;
  return true;
}

}  // namespace serialization
}  // namespace tensorstore
