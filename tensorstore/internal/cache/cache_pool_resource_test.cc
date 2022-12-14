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

#include "tensorstore/internal/cache/cache_pool_resource.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::CachePoolResource;

TEST(CachePoolResourceTest, Default) {
  auto resource_spec = Context::Resource<CachePoolResource>::DefaultSpec();
  auto cache = Context::Default().GetResource(resource_spec).value();
  EXPECT_EQ(0u, (*cache)->limits().total_bytes_limit);
  EXPECT_EQ(0u, (*cache)->limits().queued_for_writeback_bytes_limit);
}

TEST(CachePoolResourceTest, EmptyObject) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<CachePoolResource>::FromJson(
                              ::nlohmann::json::object_t{}));
  auto cache = Context::Default().GetResource(resource_spec).value();
  EXPECT_EQ(0u, (*cache)->limits().total_bytes_limit);
  EXPECT_EQ(0u, (*cache)->limits().queued_for_writeback_bytes_limit);
}

TEST(CachePoolResourceTest, TotalBytesLimitOnly) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<CachePoolResource>::FromJson(
                              {{"total_bytes_limit", 100}}));
  auto cache = Context::Default().GetResource(resource_spec).value();
  EXPECT_EQ(100u, (*cache)->limits().total_bytes_limit);
  EXPECT_EQ(50u, (*cache)->limits().queued_for_writeback_bytes_limit);
}

TEST(CachePoolResourceTest, Both) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<CachePoolResource>::FromJson(
                              {{"total_bytes_limit", 100},
                               {"queued_for_writeback_bytes_limit", 30}}));
  auto cache = Context::Default().GetResource(resource_spec).value();
  EXPECT_EQ(100u, (*cache)->limits().total_bytes_limit);
  EXPECT_EQ(30u, (*cache)->limits().queued_for_writeback_bytes_limit);
}

TEST(CachePoolResourceTest, BothEqual) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto resource_spec, Context::Resource<CachePoolResource>::FromJson(
                              {{"total_bytes_limit", 100},
                               {"queued_for_writeback_bytes_limit", 100}}));
  auto cache = Context::Default().GetResource(resource_spec).value();
  EXPECT_EQ(100u, (*cache)->limits().total_bytes_limit);
  EXPECT_EQ(100u, (*cache)->limits().queued_for_writeback_bytes_limit);
}

TEST(CachePoolResourceTest, OutOfRange) {
  auto resource_spec = Context::Resource<CachePoolResource>::FromJson(
      {{"total_bytes_limit", 100}, {"queued_for_writeback_bytes_limit", 101}});
  EXPECT_THAT(resource_spec, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
