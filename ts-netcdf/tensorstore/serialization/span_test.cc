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

#include "tensorstore/serialization/span.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/batch.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::span;
using ::tensorstore::serialization::DecodeBatch;
using ::tensorstore::serialization::EncodeBatch;

TEST(SpanSerializationTest, StaticExtent) {
  int values[2] = {1, 2};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(span<int, 2>(values)));

  int values_decoded[2] = {0, 0};
  TENSORSTORE_ASSERT_OK(DecodeBatch(encoded, span<int, 2>(values_decoded)));
  EXPECT_THAT(values_decoded, ::testing::ElementsAre(1, 2));
}

TEST(SpanSerializationTest, DynamicExtent) {
  int values[2] = {1, 2};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(span<int>(values)));

  int values_decoded[2] = {0, 0};
  TENSORSTORE_ASSERT_OK(DecodeBatch(encoded, span<int>(values_decoded)));
  EXPECT_THAT(values_decoded, ::testing::ElementsAre(1, 2));
}

}  // namespace
