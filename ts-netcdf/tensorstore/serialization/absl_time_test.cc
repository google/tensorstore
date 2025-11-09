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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"

namespace {

using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(DurationTest, SerializationRoundTrip) {
  TestSerializationRoundTrip(absl::Seconds(1));
  TestSerializationRoundTrip(-absl::Seconds(1));
  TestSerializationRoundTrip(absl::Seconds(10));
  TestSerializationRoundTrip(-absl::Seconds(10));
  TestSerializationRoundTrip(absl::Nanoseconds(5000000000));
  TestSerializationRoundTrip(-absl::Nanoseconds(5000000000));
  TestSerializationRoundTrip(absl::InfiniteDuration());
  TestSerializationRoundTrip(-absl::InfiniteDuration());
}

TEST(TimeTest, SerializationRoundTrip) {
  TestSerializationRoundTrip(absl::Seconds(1) + absl::UnixEpoch());
  TestSerializationRoundTrip(-absl::Seconds(1) + absl::UnixEpoch());
  TestSerializationRoundTrip(absl::Seconds(10) + absl::UnixEpoch());
  TestSerializationRoundTrip(-absl::Seconds(10) + absl::UnixEpoch());
  TestSerializationRoundTrip(absl::Nanoseconds(5000000000) + absl::UnixEpoch());
  TestSerializationRoundTrip(-absl::Nanoseconds(5000000000) +
                             absl::UnixEpoch());
  TestSerializationRoundTrip(absl::InfiniteFuture());
  TestSerializationRoundTrip(absl::InfinitePast());
}

}  // namespace
