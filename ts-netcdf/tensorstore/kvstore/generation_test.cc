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

#include "tensorstore/kvstore/generation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/serialization/test_util.h"

namespace {

using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(StorageGenerationTest, Basic) {
  EXPECT_TRUE(StorageGeneration::IsUnknown(StorageGeneration::Unknown()));
  EXPECT_FALSE(StorageGeneration::IsUnknown(StorageGeneration::NoValue()));

  EXPECT_FALSE(StorageGeneration::IsNoValue(StorageGeneration::Unknown()));
  EXPECT_TRUE(StorageGeneration::IsNoValue(StorageGeneration::NoValue()));
  EXPECT_TRUE(StorageGeneration::IsClean(StorageGeneration::NoValue()));
  EXPECT_FALSE(StorageGeneration::IsClean(StorageGeneration::Unknown()));
  EXPECT_EQ(StorageGeneration::NoValue(),
            StorageGeneration::Clean(StorageGeneration::NoValue()));
}

TEST(StorageGenerationTest, DebugString) {
  EXPECT_THAT(StorageGeneration::Unknown().DebugString(), "Unknown");
  EXPECT_THAT(StorageGeneration::NoValue().DebugString(), "NoValue");
  EXPECT_THAT(StorageGeneration::Invalid().DebugString(),
              ::testing::StartsWith("invalid:"));
  EXPECT_THAT(
      StorageGeneration::Dirty(StorageGeneration::Unknown(), 42).DebugString(),
      "M42+Unknown");
  EXPECT_THAT(
      StorageGeneration::Dirty(
          StorageGeneration::Dirty(StorageGeneration::Unknown(), 42), 43)
          .DebugString(),
      "M43+M42+Unknown");
  EXPECT_THAT(
      StorageGeneration::Dirty(StorageGeneration::NoValue(), 42).DebugString(),
      "M42+NoValue");
  EXPECT_THAT(StorageGeneration::AddLayer(
                  StorageGeneration::Dirty(StorageGeneration::NoValue(), 42))
                  .DebugString(),
              "|M42+NoValue");
  EXPECT_THAT(StorageGeneration::Dirty(
                  StorageGeneration::AddLayer(StorageGeneration::Dirty(
                      StorageGeneration::NoValue(), 42)),
                  43)
                  .DebugString(),
              "M43+|M42+NoValue");
}

TEST(StorageGenerationTest, StripTag) {
  for (auto base_generation :
       {StorageGeneration::Unknown(), StorageGeneration::NoValue(),
        StorageGeneration::FromString("abc")}) {
    SCOPED_TRACE("base_generation=" + base_generation.DebugString());
    auto g = StorageGeneration::Dirty(base_generation, 1);
    EXPECT_THAT(g.DebugString(), "M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripTag(g), base_generation);
    EXPECT_THAT(StorageGeneration::Clean(g), base_generation);

    auto g2 = StorageGeneration::Dirty(g, 2);
    EXPECT_THAT(g2.DebugString(), "M2+M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripTag(g2), g);
    EXPECT_THAT(StorageGeneration::Clean(g2), base_generation);
  }
}

TEST(StorageGenerationTest, StripLayer) {
  for (auto base_generation :
       {StorageGeneration::Unknown(), StorageGeneration::NoValue(),
        StorageGeneration::FromString("abc")}) {
    SCOPED_TRACE("base_generation=" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::AddLayer(base_generation).DebugString(),
                base_generation.DebugString());
    auto g = StorageGeneration::Dirty(base_generation, 1);
    EXPECT_THAT(StorageGeneration::StripLayer(g), base_generation);

    auto g2 = StorageGeneration::Dirty(g, 2);
    EXPECT_THAT(StorageGeneration::StripLayer(g2), base_generation);

    auto g3 = StorageGeneration::AddLayer(g2);
    EXPECT_EQ(g3, g2);
    EXPECT_THAT(g3.DebugString(), "|M2+M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripLayer(g3).DebugString(),
                g2.DebugString());

    auto g4 = StorageGeneration::Dirty(g3, 3);
    EXPECT_THAT(g4.DebugString(), "M3+|M2+M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripLayer(g4), g2);

    auto g5 = StorageGeneration::Dirty(g4, 4);
    EXPECT_THAT(g5.DebugString(),
                "M4+M3+|M2+M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripLayer(g5), g2);

    auto g6 = StorageGeneration::AddLayer(g5);
    EXPECT_THAT(g6.DebugString(),
                "|M4+M3+|M2+M1+" + base_generation.DebugString());
    EXPECT_THAT(StorageGeneration::StripLayer(g6).DebugString(),
                g5.DebugString());
    EXPECT_THAT(StorageGeneration::Clean(g6), base_generation);
  }
}

TEST(StorageGenerationTest, LastMutatedBy) {
  EXPECT_TRUE(StorageGeneration::Dirty(StorageGeneration::Unknown(), 1)
                  .LastMutatedBy(1));
  EXPECT_FALSE(StorageGeneration::Dirty(StorageGeneration::Unknown(), 1)
                   .LastMutatedBy(0));
  EXPECT_FALSE(StorageGeneration::Unknown().LastMutatedBy(0));
  EXPECT_FALSE(StorageGeneration::NoValue().LastMutatedBy(0));
}

TEST(StorageGenerationTest, Condition) {
  EXPECT_THAT(StorageGeneration::Condition(StorageGeneration::Unknown(),
                                           StorageGeneration::Unknown()),
              StorageGeneration::Unknown());
  EXPECT_THAT(StorageGeneration::Condition(StorageGeneration::Unknown(),
                                           StorageGeneration::NoValue()),
              StorageGeneration::NoValue());
  EXPECT_THAT(StorageGeneration::Condition(StorageGeneration::FromString("abc"),
                                           StorageGeneration::NoValue()),
              StorageGeneration::FromString("abc"));
  EXPECT_THAT(
      StorageGeneration::Condition(
          StorageGeneration::Unknown(),
          StorageGeneration::Dirty(StorageGeneration::FromString("abc"), 1)),
      StorageGeneration::FromString("abc"));
  EXPECT_THAT(StorageGeneration::Condition(
                  StorageGeneration::Unknown(),
                  StorageGeneration::AddLayer(StorageGeneration::Dirty(
                      StorageGeneration::FromString("abc"), 1)))
                  .DebugString(),
              "|M1+\"abc\"");
  EXPECT_THAT(StorageGeneration::Condition(
                  StorageGeneration::Dirty(StorageGeneration::Unknown(), 2),
                  StorageGeneration::AddLayer(StorageGeneration::Dirty(
                      StorageGeneration::FromString("abc"), 1)))
                  .DebugString(),
              "M2+|M1+\"abc\"");
  EXPECT_THAT(StorageGeneration::Condition(
                  StorageGeneration::Dirty(StorageGeneration::Unknown(), 2),
                  StorageGeneration::Dirty(StorageGeneration::Unknown(), 1))
                  .DebugString(),
              "M2+Unknown");
}

TEST(StorageGenerationTest, Uint64) {
  auto g = StorageGeneration::FromUint64(12345);
  EXPECT_TRUE(StorageGeneration::IsUint64(g));
  EXPECT_EQ(12345, StorageGeneration::ToUint64(g));
  EXPECT_FALSE(StorageGeneration::IsUint64(StorageGeneration::Unknown()));
  EXPECT_FALSE(StorageGeneration::IsUint64(StorageGeneration::NoValue()));
  EXPECT_FALSE(StorageGeneration::IsUint64(StorageGeneration::Invalid()));
}

TEST(StorageGenerationSerializationTest, Basic) {
  TestSerializationRoundTrip(StorageGeneration::Unknown());
  TestSerializationRoundTrip(StorageGeneration::FromUint64(12345));
}

TEST(TimestampedStorageGenerationSerializationTest, Basic) {
  TestSerializationRoundTrip(TimestampedStorageGeneration(
      StorageGeneration::FromUint64(12345), absl::InfinitePast()));
  TestSerializationRoundTrip(TimestampedStorageGeneration(
      StorageGeneration::FromUint64(12345), absl::InfiniteFuture()));
}

TEST(StorageGenerationTest, IsCleanValidValue) {
  EXPECT_FALSE(
      StorageGeneration::IsCleanValidValue(StorageGeneration::Unknown()));
  EXPECT_FALSE(
      StorageGeneration::IsCleanValidValue(StorageGeneration::NoValue()));
  EXPECT_FALSE(
      StorageGeneration::IsCleanValidValue(StorageGeneration::Invalid()));
  EXPECT_TRUE(StorageGeneration::IsCleanValidValue(
      StorageGeneration::FromString("abc")));
  EXPECT_TRUE(
      StorageGeneration::IsCleanValidValue(StorageGeneration::FromUint64(42)));
}

TEST(StorageGenerationTest, DecodeString) {
  EXPECT_EQ("abc", StorageGeneration::DecodeString(
                       StorageGeneration::FromString("abc")));
}

}  // namespace
