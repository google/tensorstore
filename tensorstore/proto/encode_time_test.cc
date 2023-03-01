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

#include "google/protobuf/duration.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include <gtest/gtest.h>
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::internal::AbslDurationToProto;
using ::tensorstore::internal::AbslTimeToProto;
using ::tensorstore::internal::ProtoToAbslDuration;
using ::tensorstore::internal::ProtoToAbslTime;

TEST(EncodeTimestamp, Basic) {
  auto roundtrip = [](absl::Time ts) {
    google::protobuf::Timestamp proto;
    AbslTimeToProto(ts, &proto);
    return ProtoToAbslTime(proto);
  };

  tensorstore::Result<absl::Time> result;

  result = roundtrip(absl::InfinitePast());
  TENSORSTORE_ASSERT_OK(result);
  EXPECT_EQ(absl::InfinitePast(), *result);

  result = roundtrip(absl::InfiniteFuture());
  TENSORSTORE_ASSERT_OK(result);
  EXPECT_EQ(absl::InfiniteFuture(), *result);

  auto now = absl::Now();
  result = roundtrip(now);
  TENSORSTORE_ASSERT_OK(result);
  EXPECT_EQ(now, *result);
}

TEST(EncodeDuration, Basic) {
  auto roundtrip = [](absl::Duration d) {
    google::protobuf::Duration proto;
    AbslDurationToProto(d, &proto);
    return ProtoToAbslDuration(proto);
  };

  auto test_roundtrip = [&](absl::Duration d) {
    SCOPED_TRACE(tensorstore::StrCat("duration=", d));
    EXPECT_THAT(roundtrip(d), ::testing::Optional(d));
  };

  test_roundtrip(absl::InfiniteDuration());
  test_roundtrip(-absl::InfiniteDuration());
  test_roundtrip(absl::Seconds(5));
  test_roundtrip(absl::Seconds(-5));
  test_roundtrip(absl::ZeroDuration());
  test_roundtrip(absl::Milliseconds(12345));
  test_roundtrip(absl::Milliseconds(-12345));
}

}  // namespace
