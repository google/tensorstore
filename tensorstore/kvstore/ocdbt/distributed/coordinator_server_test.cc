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

#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::KeyRange;
using ::tensorstore::internal_ocdbt::BtreeNodeIdentifier;
using ::tensorstore::internal_ocdbt_cooperator::LeaseCacheForCooperator;
using ::tensorstore::ocdbt::CoordinatorServer;

class CoordinatorServerTest : public ::testing::Test {
 protected:
  absl::Time cur_time;
  CoordinatorServer server_;
  LeaseCacheForCooperator lease_cache;

  void SetUp() override {
    auto security =
        ::tensorstore::internal_ocdbt::GetInsecureRpcSecurityMethod();
    CoordinatorServer::Options options;
    options.spec.security = security;
    options.spec.bind_addresses.push_back("localhost:0");
    options.clock = [this] { return cur_time; };
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        server_, CoordinatorServer::Start(std::move(options)));

    auto channel =
        grpc::CreateChannel(tensorstore::StrCat("localhost:", server_.port()),
                            security->GetClientCredentials());
    LeaseCacheForCooperator::Options lease_cache_options;
    lease_cache_options.clock = {};
    lease_cache_options.cooperator_port = 42;
    lease_cache_options.coordinator_stub =
        tensorstore::internal_ocdbt::grpc_gen::Coordinator::NewStub(channel);
    lease_cache_options.security = security;

    lease_cache = LeaseCacheForCooperator(std::move(lease_cache_options));
  }
};

TEST_F(CoordinatorServerTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto lease_info,
      lease_cache
          .GetLease("key", BtreeNodeIdentifier{1, KeyRange{"abc", "def"}})
          .result());
  EXPECT_FALSE(lease_info->peer_stub);
  EXPECT_THAT(lease_info->peer_address, ::testing::MatchesRegex(".*:42"));
}

}  // namespace
