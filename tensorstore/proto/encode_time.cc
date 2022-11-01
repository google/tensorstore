// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/proto/encode_time.h"

#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

void AbslTimeToProto(absl::Time t, google::protobuf::Timestamp* proto) {
  if (t == absl::InfiniteFuture()) {
    proto->set_seconds(0x7FFFFFFFFFFFFFFFll);
    proto->set_nanos(0);
  } else if (t == absl::InfinitePast()) {
    proto->set_seconds(0x8000000000000000ll);
    proto->set_nanos(0);
  } else {
    const int64_t s = absl::ToUnixSeconds(t);
    const int64_t n = (t - absl::FromUnixSeconds(s)) / absl::Nanoseconds(1);
    proto->set_seconds(s);
    proto->set_nanos(n);
  }
}

tensorstore::Result<absl::Time> ProtoToAbslTime(
    const google::protobuf::Timestamp& proto) {
  const auto sec = proto.seconds();
  const auto ns = proto.nanos();
  // Interpret sintinels as positive/negative infinity.
  if (sec == 0x7FFFFFFFFFFFFFFFll) {
    return absl::InfiniteFuture();
  }
  if (sec == 0x8000000000000000ll) {
    return absl::InfinitePast();
  }
  // Otherwise validate according to: google/protobuf/timestamp.proto
  // sec must be [0001-01-01T00:00:00Z, 9999-12-31T23:59:59.999999999Z]
  if (sec < -62135596800 || sec > 253402300799) {
    return absl::InvalidArgumentError(tensorstore::StrCat("seconds=", sec));
  }
  if (ns < 0 || ns > 999999999) {
    return absl::InvalidArgumentError(tensorstore::StrCat("nanos=", ns));
  }
  return absl::FromUnixSeconds(sec) + absl::Nanoseconds(ns);
}

}  // namespace internal
}  // namespace tensorstore
