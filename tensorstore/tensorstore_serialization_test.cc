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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_tuple.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::serialization::SerializationRoundTrip;

TEST(TensorStoreSerializationTest, Invalid) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto invalid_store,
      SerializationRoundTrip(tensorstore::TensorStore<int, 2>()));
  EXPECT_FALSE(invalid_store.valid());
}

TEST(TensorStoreSerializationTest, Simple) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}})
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_store,
                                   SerializationRoundTrip(store));
  EXPECT_THAT(tensorstore::Read(decoded_store).result(),
              ::testing::Optional(tensorstore::MakeArray<int32_t>({1, 2, 3})));
}

TEST(TensorStoreSerializationTest, SharedContext) {
  tensorstore::internal::ScopedTemporaryDirectory temp_dir;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      tensorstore::Spec::FromJson({
          {"driver", "n5"},
          {"kvstore", {{"driver", "file"}, {"path", temp_dir.path()}}},
          {"metadata",
           {{"compression", {{"type", "raw"}}},
            {"dataType", "uint32"},
            {"dimensions", {2}},
            {"blockSize", {2}}}},
          {"recheck_cached_data", false},
          {"recheck_cached_metadata", false},
          {"create", true},
          {"open", true},
      }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson({{"cache_pool", {{"total_bytes_limit", 1000000}}}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto t1,
                                   tensorstore::Open(spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto t2,
                                   tensorstore::Open(spec, context).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded, SerializationRoundTrip(std::make_tuple(t1, t2)));
  auto [decoded_t1, decoded_t2] = decoded;
  EXPECT_THAT(tensorstore::Read(decoded_t1).result(),
              ::testing::Optional(tensorstore::MakeArray<uint32_t>({0, 0})));
  EXPECT_THAT(tensorstore::Read(decoded_t2).result(),
              ::testing::Optional(tensorstore::MakeArray<uint32_t>({0, 0})));
  TENSORSTORE_ASSERT_OK(tensorstore::Write(
      tensorstore::MakeScalarArray<uint32_t>(42), decoded_t1));

  // Delete data
  TENSORSTORE_ASSERT_OK(tensorstore::Open(
      spec,
      tensorstore::OpenMode::create | tensorstore::OpenMode::delete_existing));

  // decoded_t1 still sees old data in cache
  EXPECT_THAT(tensorstore::Read(decoded_t1).result(),
              ::testing::Optional(tensorstore::MakeArray<uint32_t>({42, 42})));

  // decoded_t2 shares cache with decoded_t1
  EXPECT_THAT(tensorstore::Read(decoded_t2).result(),
              ::testing::Optional(tensorstore::MakeArray<uint32_t>({42, 42})));
}

}  // namespace
