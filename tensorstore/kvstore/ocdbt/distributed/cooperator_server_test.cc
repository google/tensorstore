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

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/clientauth/create_channel.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/cooperator.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator.grpc.pb.h"
#include "tensorstore/kvstore/ocdbt/distributed/coordinator_server.h"
#include "tensorstore/kvstore/ocdbt/distributed/lease_cache_for_cooperator.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::KeyRange;
using ::tensorstore::KvStore;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::CachePool;
using ::tensorstore::internal::CachePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal_ocdbt::BtreeInteriorNodeWriteMutation;
using ::tensorstore::internal_ocdbt::BtreeLeafNodeWriteMutation;
using ::tensorstore::internal_ocdbt::BtreeNodeIdentifier;
using ::tensorstore::internal_ocdbt::BtreeNodeWriteMutation;
using ::tensorstore::internal_ocdbt::ConfigConstraints;
using ::tensorstore::internal_ocdbt::ConfigState;
using ::tensorstore::internal_ocdbt::IoHandle;
using ::tensorstore::internal_ocdbt_cooperator::LeaseCacheForCooperator;
using ::tensorstore::ocdbt::CoordinatorServer;
using ::testing::HasSubstr;

namespace internal_ocdbt_cooperator = ::tensorstore::internal_ocdbt_cooperator;

class HungCooperatorService : public tensorstore::internal_ocdbt::grpc_gen::
                                  Cooperator::CallbackService {
 public:
  grpc::ServerUnaryReactor* GetOrCreateManifest(
      grpc::CallbackServerContext* context,
      const tensorstore::internal_ocdbt::grpc_gen::GetOrCreateManifestRequest*
          request,
      tensorstore::internal_ocdbt::grpc_gen::GetOrCreateManifestResponse*
          response) override {
    return HoldReactor(context);
  }

  grpc::ServerUnaryReactor* Write(
      grpc::CallbackServerContext* context,
      const tensorstore::internal_ocdbt::grpc_gen::WriteRequest* request,
      tensorstore::internal_ocdbt::grpc_gen::WriteResponse* response) override {
    return HoldReactor(context);
  }

  void FinishAllPending() {
    std::vector<grpc::ServerUnaryReactor*> to_finish;
    {
      absl::MutexLock lock(mutex_);
      shutting_down_ = true;
      to_finish.swap(pending_reactors_);
    }
    for (auto* reactor : to_finish) {
      reactor->Finish(grpc::Status::CANCELLED);
    }
  }

  int calls() const {
    absl::MutexLock lock(mutex_);
    return calls_;
  }

 private:
  grpc::ServerUnaryReactor* HoldReactor(grpc::CallbackServerContext* context) {
    auto* reactor = context->DefaultReactor();
    absl::MutexLock lock(mutex_);
    ++calls_;
    if (shutting_down_) {
      reactor->Finish(grpc::Status::CANCELLED);
    } else {
      pending_reactors_.push_back(reactor);
    }
    return reactor;
  }

  mutable absl::Mutex mutex_;
  bool shutting_down_ = false;
  int calls_ = 0;
  std::vector<grpc::ServerUnaryReactor*> pending_reactors_;
};

class CooperatorServerTest : public ::testing::Test {
 protected:
  tensorstore::KvStore base_kvstore_;
  IoHandle::Ptr io_handle_;
  CoordinatorServer coordinator_server_;
  internal_ocdbt_cooperator::CooperatorPtr cooperator_;

  CooperatorServerTest() {
    auto security =
        ::tensorstore::internal_ocdbt::GetInsecureRpcSecurityMethod();
    auto cache_pool = CachePool::Make({});
    auto data_copy_concurrency =
        Context::Default()
            .GetResource<tensorstore::internal::DataCopyConcurrencyResource>()
            .value();
    base_kvstore_ = tensorstore::GetMemoryKeyValueStore();
    io_handle_ = MakeIoHandle(
        data_copy_concurrency, cache_pool.get(), base_kvstore_, base_kvstore_,
        ConfigState::Make(ConfigConstraints{},
                          base_kvstore_.driver->GetSupportedFeatures({}))
            .value(),
        /*data_file_prefixes=*/{});

    {
      CoordinatorServer::Options options;
      options.spec.security = security;
      options.spec.bind_addresses.push_back("localhost:0");
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          coordinator_server_, CoordinatorServer::Start(std::move(options)));
    }

    std::string coordinator_address =
        absl::StrCat("localhost:", coordinator_server_.port());

    {
      internal_ocdbt_cooperator::Options options;
      options.io_handle = io_handle_;
      options.bind_addresses.push_back("localhost:0");
      options.coordinator_address = coordinator_address;
      options.security = security;
      options.lease_duration = absl::Milliseconds(200);
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          cooperator_, internal_ocdbt_cooperator::Start(std::move(options)));
    }
  }
};

TEST_F(CooperatorServerTest, Basic) {
  TENSORSTORE_CHECK_OK(internal_ocdbt_cooperator::GetManifestForWriting(
      *cooperator_, absl::InfinitePast()));
}

// Submits a single-mutation batch for the leased node `identifier`.
tensorstore::Result<internal_ocdbt_cooperator::MutationBatchResponse>
SubmitMutation(internal_ocdbt_cooperator::Cooperator& cooperator,
               BtreeNodeIdentifier identifier,
               BtreeNodeWriteMutation::Ptr mutation, uint64_t root_generation,
               StorageGeneration node_generation) {
  internal_ocdbt_cooperator::MutationBatchRequest batch_request;
  batch_request.root_generation = root_generation;
  batch_request.node_generation = std::move(node_generation);
  batch_request.mutations.push_back(
      {std::move(mutation), tensorstore::MakeReadyFuture()});
  return internal_ocdbt_cooperator::SubmitMutationBatch(
             cooperator, std::move(identifier), std::move(batch_request))
      .result();
}

BtreeNodeWriteMutation::Ptr MakeInteriorDeleteMutation() {
  auto mutation = MakeIntrusivePtr<BtreeInteriorNodeWriteMutation>();
  mutation->mode = BtreeNodeWriteMutation::kDeleteExisting;
  mutation->existing_range = KeyRange::Prefix("a");
  mutation->existing_generation = StorageGeneration::NoValue();
  return mutation;
}

BtreeNodeWriteMutation::Ptr MakeLeafWriteMutation(std::string key) {
  auto mutation = MakeIntrusivePtr<BtreeLeafNodeWriteMutation>();
  mutation->mode = BtreeNodeWriteMutation::kAddNew;
  mutation->key = std::move(key);
  mutation->existing_generation = StorageGeneration::NoValue();
  mutation->new_entry.value_reference = absl::Cord("v");
  return mutation;
}

// Submitting an interior-node mutation (`identifier.height > 0`) when the
// root B-tree is empty (`latest_version.root.location.IsMissing()`) must call
// `LeasedNodeGone()` rather than downcasting `BtreeInteriorNodeWriteMutation`
// to `BtreeLeafNodeWriteMutation` (undefined behavior / type confusion).
TEST_F(CooperatorServerTest, HeightMismatchEmptyRoot) {
  TENSORSTORE_ASSERT_OK(internal_ocdbt_cooperator::GetManifestForWriting(
      *cooperator_, absl::InfinitePast()));
  EXPECT_THAT(SubmitMutation(
                  *cooperator_, BtreeNodeIdentifier{/*height=*/1, KeyRange{}},
                  MakeInteriorDeleteMutation(),
                  /*root_generation=*/1, StorageGeneration::NoValue()),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Leased node no longer exists")));
}

class CooperatorServerLeafRootTest : public CooperatorServerTest {
 protected:
  // Commits a single leaf entry so that the root is a non-empty height-0 node
  // at generation 2.
  void SetUp() override {
    TENSORSTORE_ASSERT_OK(internal_ocdbt_cooperator::GetManifestForWriting(
        *cooperator_, absl::InfinitePast()));
    TENSORSTORE_ASSERT_OK(SubmitMutation(
        *cooperator_, BtreeNodeIdentifier{/*height=*/0, KeyRange{}},
        MakeLeafWriteMutation("a"), /*root_generation=*/1,
        StorageGeneration::NoValue()));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto manifest_with_time,
        io_handle_->GetManifest(absl::InfiniteFuture()).result());
    ASSERT_THAT(manifest_with_time.manifest, ::testing::NotNull());
    const auto& manifest = *manifest_with_time.manifest;
    ASSERT_EQ(manifest.latest_generation(), 2);
    ASSERT_EQ(manifest.latest_version().root_height, 0);
    root_node_generation_ = StorageGeneration::FromString(
        manifest.latest_version().root.location.file_id.FullPath());
  }

  StorageGeneration root_node_generation_;
};

// Submitting an interior-node mutation (`identifier.height == 1`) when the
// current root is a non-empty leaf (`root_height == 0`) must call
// `LeasedNodeGone()` rather than calling `std::get<InteriorNodeEntries>` on a
// leaf node (which throws `std::bad_variant_access` / aborts).
TEST_F(CooperatorServerLeafRootTest, LeasedNodeAboveRoot) {
  EXPECT_THAT(SubmitMutation(*cooperator_,
                             BtreeNodeIdentifier{/*height=*/1, KeyRange{}},
                             MakeInteriorDeleteMutation(),
                             /*root_generation=*/2, root_node_generation_),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Leased node no longer exists")));
}

// A leased node at the root height whose range is not the full range cannot be
// the root, and must call `LeasedNodeGone()` rather than submitting the update
// to a non-existent parent at height `root_height + 1` (which wraps to 0 when
// `root_height == 255`).
TEST_F(CooperatorServerLeafRootTest, LeasedNodeAtRootHeightWithPartialRange) {
  EXPECT_THAT(
      SubmitMutation(*cooperator_,
                     BtreeNodeIdentifier{/*height=*/0, KeyRange("a", "b")},
                     MakeLeafWriteMutation("a"),
                     /*root_generation=*/2, root_node_generation_),
      StatusIs(absl::StatusCode::kAborted,
               HasSubstr("Leased node no longer exists")));
}

TEST_F(CooperatorServerTest, GrayFailurePeerTimeoutTakesOverLease) {
  // First create the manifest on the healthy cooperator.
  TENSORSTORE_ASSERT_OK(internal_ocdbt_cooperator::GetManifestForWriting(
                            *cooperator_, absl::InfinitePast())
                            .result());

  // Start a hung peer cooperator service that never responds to Write RPCs.
  HungCooperatorService hung_service;
  int hung_port = 0;
  grpc::ServerBuilder builder;
  builder.RegisterService(&hung_service);
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &hung_port);
  std::unique_ptr<grpc::Server> hung_server = builder.BuildAndStart();

  // Have the hung peer steal the lease for a leaf subtree node with a long
  // (10-second) lease duration so passive expiration cannot rescue the write.
  auto security = ::tensorstore::internal_ocdbt::GetInsecureRpcSecurityMethod();
  auto auth_strategy = security->GetClientAuthenticationStrategy();
  std::string coordinator_address =
      absl::StrCat("localhost:", coordinator_server_.port());
  grpc::ChannelArguments args;
  auto channel = ::tensorstore::internal_grpc::CreateChannel(
      *auth_strategy, coordinator_address, args);

  LeaseCacheForCooperator::Options hung_cache_options;
  hung_cache_options.clock = [] { return absl::Now(); };
  hung_cache_options.cooperator_port = hung_port;
  hung_cache_options.auth_strategy = auth_strategy;
  hung_cache_options.coordinator_stub =
      tensorstore::internal_ocdbt::grpc_gen::Coordinator::NewStub(channel);
  hung_cache_options.lease_duration = absl::Seconds(10);
  LeaseCacheForCooperator hung_lease_cache(std::move(hung_cache_options));

  auto node_id = BtreeNodeIdentifier::Root();
  auto key = node_id.GetKey("");
  // Acquire the lease on the coordinator on behalf of the hung peer.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto existing_lease, hung_lease_cache.GetLease(key, node_id).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto hung_lease,
      hung_lease_cache
          .GetLease(key, node_id, /*uncooperative_lease=*/existing_lease.get())
          .result());
  ASSERT_FALSE(hung_lease->peer_stub);

  // Start a second healthy cooperator with a 200ms lease_duration (and no
  // cached lease yet) and submit a mutation batch for the root node.
  internal_ocdbt_cooperator::Options healthy_options;
  healthy_options.io_handle = io_handle_;
  healthy_options.bind_addresses.push_back("localhost:0");
  healthy_options.coordinator_address = coordinator_address;
  healthy_options.security = security;
  healthy_options.lease_duration = absl::Milliseconds(200);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto healthy_cooperator,
      internal_ocdbt_cooperator::Start(std::move(healthy_options)));

  internal_ocdbt_cooperator::MutationBatchRequest batch_request;
  batch_request.root_generation = 1;
  auto mutation = MakeIntrusivePtr<BtreeLeafNodeWriteMutation>();
  mutation->mode = BtreeNodeWriteMutation::kDeleteExisting;
  mutation->key = "test_key";
  batch_request.mutations.push_back({std::move(mutation), {}});

  auto write_future = internal_ocdbt_cooperator::SubmitMutationBatch(
      *healthy_cooperator, BtreeNodeIdentifier::Root(),
      std::move(batch_request));

  // The healthy cooperator must time out the hung peer's Write RPC after
  // lease_duration (200ms), revoke the hung peer's lease via
  // uncooperative_lease_id, and complete the mutation batch locally.
  bool completed_in_time = write_future.WaitFor(absl::Seconds(2));
  EXPECT_TRUE(completed_in_time);
  if (completed_in_time) {
    TENSORSTORE_EXPECT_OK(write_future.result());
    EXPECT_GE(hung_service.calls(), 1);
  }

  hung_service.FinishAllPending();
  hung_server->Shutdown();
  hung_server->Wait();
}

}  // namespace
