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

#include "tensorstore/serialization/status.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"

namespace {

using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(StatusTest, SimpleRoundTrip) {
  TestSerializationRoundTrip(absl::OkStatus());
  TestSerializationRoundTrip(absl::InvalidArgumentError("abc"));
}

TEST(StatusTest, PayloadRoundTrip) {
  auto status = absl::InternalError("xyz");
  status.SetPayload("a", absl::Cord("b"));
  TestSerializationRoundTrip(status);
  status.SetPayload("c", absl::Cord("d"));
  TestSerializationRoundTrip(status);
}

}  // namespace
