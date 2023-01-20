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

#include "tensorstore/kvstore/grpc/kvstore_service.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpc/grpc_security_constants.h"
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/grpc/grpc_kvstore.h"
#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::KeyRange;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore_grpc::KvStoreServiceImpl;
using ::tensorstore_grpc::kvstore::grpc_gen::KvStoreService;

// This test uses the memory kvstore driver over grpc, so the test
// assertions should closely mimic those in the memory kvstore driver.
class KvStoreTest : public testing::Test {
 public:
  KvStoreTest()
      : service_(std::make_shared<KvStoreServiceImpl>(
            tensorstore::kvstore::Open({{"driver", "memory"}},
                                       tensorstore::Context::Default())
                .result()
                .value())),
        server_(ABSL_DIE_IF_NULL(
            ::grpc::ServerBuilder()
                .AddListeningPort(
                    "[::]:0",
                    ::grpc::experimental::LocalServerCredentials(LOCAL_TCP),
                    &listening_port_)
                .RegisterService(service_.get())
                .BuildAndStart())) {}

  ~KvStoreTest() {
    server_->Shutdown();
    server_->Wait();
  }

  // Creates a stub that communicates with the mock server using local
  // credentials.
  std::unique_ptr<KvStoreService::StubInterface> NewStub() {
    return KvStoreService::NewStub(
        server_->InProcessChannel(grpc::ChannelArguments()));
  }

  kvstore::DriverPtr OpenStore() {
    auto store_result = tensorstore::CreateGrpcKvStore(
        absl::StrCat("[::]:", listening_port_), absl::Seconds(10), NewStub());
    EXPECT_TRUE(store_result.ok()) << store_result.status();
    return std::move(store_result).value();
  }

 private:
  int listening_port_;
  std::shared_ptr<tensorstore_grpc::KvStoreServiceImpl> service_;
  const std::unique_ptr<::grpc::Server> server_;
};

TEST_F(KvStoreTest, Basic) {
  auto store = OpenStore();
  tensorstore::internal::TestKeyValueStoreBasicFunctionality(store);
}

TEST_F(KvStoreTest, DeleteRange) {
  auto store = OpenStore();

  TENSORSTORE_EXPECT_OK(store->Write("a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/f", absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(store->DeleteRange(KeyRange::Prefix("a/c")));

  EXPECT_EQ("xyz", store->Read("a/b").value().value);
  EXPECT_EQ("xyz", store->Read("a/d").value().value);

  EXPECT_THAT(store->Read("a/c/x").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/y").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/e").result(), MatchesKvsReadResultNotFound());
  EXPECT_THAT(store->Read("a/c/z/f").result(), MatchesKvsReadResultNotFound());
}

TEST_F(KvStoreTest, List) {
  auto store = OpenStore();

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}),
                                   tensorstore::LoggingReceiver{&log});
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  TENSORSTORE_EXPECT_OK(store->Write("a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(store->Write("a/c/z/f", absl::Cord("xyz")));

  // Listing the entire stream works.
  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}),
                                   tensorstore::LoggingReceiver{&log});

    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing a subset of the stream works.
  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({KeyRange::Prefix("a/c/")}),
                                   tensorstore::LoggingReceiver{&log});

    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: a/c/z/f",
                         "set_value: a/c/y", "set_value: a/c/z/e",
                         "set_value: a/c/x", "set_done", "set_stopping"));
  }

  // Cancellation immediately after starting yields nothing..
  struct CancelOnStarting : public tensorstore::LoggingReceiver {
    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->tensorstore::LoggingReceiver::set_starting({});
      do_cancel();
    }
  };

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}), CancelOnStarting{{&log}});

    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  struct CancelAfter2 : public tensorstore::LoggingReceiver {
    using Key = tensorstore::kvstore::Key;
    tensorstore::AnyCancelReceiver cancel;

    void set_starting(tensorstore::AnyCancelReceiver do_cancel) {
      this->cancel = std::move(do_cancel);
      this->tensorstore::LoggingReceiver::set_starting({});
    }

    void set_value(Key k) {
      this->tensorstore::LoggingReceiver::set_value(std::move(k));
      if (this->log->size() == 2) {
        this->cancel();
      }
    }
  };

  {
    std::vector<std::string> log;
    tensorstore::execution::submit(store->List({}), CancelAfter2{{&log}});

    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    ::testing::AnyOf("set_value: a/d", "set_value: a/c/z/f",
                                     "set_value: a/c/y", "set_value: a/c/z/e",
                                     "set_value: a/c/x", "set_value: a/b"),
                    "set_done", "set_stopping"));
  }
}

}  // namespace
