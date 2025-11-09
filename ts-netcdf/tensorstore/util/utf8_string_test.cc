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

#include "tensorstore/util/utf8_string.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::Utf8String;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(SerializationTest, Valid) {
  TestSerializationRoundTrip(Utf8String{""});
  TestSerializationRoundTrip(Utf8String{"abc"});
  TestSerializationRoundTrip(Utf8String{"\xc2\x80hello\xc2\xbf"});
}

TEST(SerializationTest, Invalid) {
  EXPECT_THAT(SerializationRoundTrip(Utf8String{"\xC1"}),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            "String is not valid utf-8: .*"));
}

}  // namespace
