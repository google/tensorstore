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

#include <gtest/gtest.h>
#include "tensorstore/serialization/serialization.h"
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

  EXPECT_EQ(StorageGeneration{std::string{StorageGeneration::kDirty}},
            StorageGeneration::Dirty(StorageGeneration::Unknown()));

  StorageGeneration gen{
      std::string{1, 2, 3, 4, 5, StorageGeneration::kBaseGeneration}};
  StorageGeneration local_gen{std::string{
      1, 2, 3, 4, 5,
      StorageGeneration::kBaseGeneration | StorageGeneration::kDirty}};
  EXPECT_FALSE(StorageGeneration::IsUnknown(gen));
  EXPECT_FALSE(StorageGeneration::IsUnknown(local_gen));
  EXPECT_TRUE(StorageGeneration::IsClean(gen));
  EXPECT_FALSE(StorageGeneration::IsClean(local_gen));
  EXPECT_FALSE(StorageGeneration::IsDirty(gen));
  EXPECT_TRUE(StorageGeneration::IsDirty(local_gen));
  EXPECT_EQ(local_gen, StorageGeneration::Dirty(gen));
  EXPECT_EQ(gen, StorageGeneration::Clean(local_gen));
  EXPECT_TRUE(StorageGeneration::IsClean(StorageGeneration::NoValue()));
  EXPECT_FALSE(StorageGeneration::IsClean(StorageGeneration::Unknown()));
  EXPECT_EQ(StorageGeneration::NoValue(),
            StorageGeneration::Clean(StorageGeneration::NoValue()));
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
