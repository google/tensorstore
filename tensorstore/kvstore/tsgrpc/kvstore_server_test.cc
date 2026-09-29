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

#include "tensorstore/kvstore/tsgrpc/kvstore_server.h"

#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.pb.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::KeyRange;
using ::tensorstore::grpc_kvstore::KvStoreServer;
using ::tensorstore::internal::IsRegularStorageGeneration;
using ::tensorstore::internal::KeyValueStoreOpsTestParameters;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal::MatchesTimestampedStorageGeneration;

class KvStoreSingleton {
 public:
  KvStoreSingleton() : ctx_(tensorstore::Context::Default()) {
    server_ = KvStoreServer::Start(KvStoreServer::Spec::FromJson(  //
                                       {
                                           {"bind_addresses", {"localhost:0"}},
                                           {"base", "memory://x"},
                                       })
                                       .value(),
                                   ctx_)
                  .value();
    address_ = absl::StrFormat("localhost:%d", server_.port());
  }

  const std::string& address() const { return address_; }

 private:
  tensorstore::Context ctx_;
  KvStoreServer server_;
  std::string address_;
};

const KvStoreSingleton& GetSingleton() {
  static const KvStoreSingleton* const kSingleton = new KvStoreSingleton();
  return *kSingleton;
}

// This test uses the memory kvstore driver over grpc, so the test
// assertions should closely mimic those in the memory kvstore driver.
class KvStoreTest : public testing::Test {
 public:
  const std::string& address() const { return GetSingleton().address(); }
};

TENSORSTORE_GLOBAL_INITIALIZER {
  KeyValueStoreOpsTestParameters params;
  params.test_name = "Basic";
  params.get_store = [](auto callback) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::kvstore::Open({{"driver", "tsgrpc_kvstore"},
                                    {"address", GetSingleton().address()},
                                    {"path", "basic/"}})
            .result());
    callback(store);
  };
  params.atomic_transaction = false;
  params.test_list_without_prefix = false;
  RegisterKeyValueStoreOpsTests(params);
}

TEST_F(KvStoreTest, DeleteRange) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open({{"driver", "tsgrpc_kvstore"},
                                              {"address", address()},
                                              {"path", "delete_range/"}},
                                             context)
                      .result());

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange::Prefix("a/c")));

  EXPECT_EQ("xyz", kvstore::Read(store, "a/b").value().value);
  EXPECT_EQ("xyz", kvstore::Read(store, "a/d").value().value);

  EXPECT_THAT(kvstore::Read(store, "a/c/x").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/y").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/e").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/f").result(),
              MatchesKvsReadResultNotFound());
}

TEST_F(KvStoreTest, List) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open({{"driver", "tsgrpc_kvstore"},
                                              {"address", address()},
                                              {"path", "list/"}},
                                             context)
                      .result());

  {
    std::vector<std::string> log;
    absl::Notification notification;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));

  // Listing the entire stream works.
  {
    std::vector<std::string> log;
    absl::Notification notification;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});

    notification.WaitForNotification();
    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre(
                 "set_starting", "set_value: a/d", "set_value: a/c/z/f",
                 "set_value: a/c/y", "set_value: a/c/z/e", "set_value: a/c/x",
                 "set_value: a/b", "set_done", "set_stopping"));
  }

  // Listing a subset of the stream works.
  {
    std::vector<std::string> log;
    absl::Notification notification;
    tensorstore::execution::submit(
        kvstore::List(store, {KeyRange::Prefix("a/c/")}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});

    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: a/c/z/f",
                         "set_value: a/c/y", "set_value: a/c/z/e",
                         "set_value: a/c/x", "set_done", "set_stopping"));
  }

  // Cancellation immediately after starting yields nothing..
  {
    std::vector<std::string> log;
    absl::Notification notification;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::CancelOnStartingReceiver{{&log}}});

    notification.WaitForNotification();
    EXPECT_THAT(log, ::testing::ElementsAre("set_starting", "set_done",
                                            "set_stopping"));
  }

  // Cancellation in the middle of the stream stops the stream.
  {
    std::vector<std::string> log;
    absl::Notification notification;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::CancelAfterNReceiver<2>{{&log}}});

    notification.WaitForNotification();
    EXPECT_THAT(log,
                ::testing::ElementsAre(
                    "set_starting",
                    ::testing::AnyOf("set_value: a/d", "set_value: a/c/z/f",
                                     "set_value: a/c/y", "set_value: a/c/z/e",
                                     "set_value: a/c/x", "set_value: a/b"),
                    "set_done", "set_stopping"));
  }
}

TEST_F(KvStoreTest, MultiPartReadWrite) {
  absl::Cord value;
  char x = ' ';
  while (value.size() < (2 << 20)) {
    value.Append(std::string(1 << 12, x));
    x++;
    if (static_cast<int>(x) > 126) x = ' ';
  }

  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::kvstore::Open({{"driver", "tsgrpc_kvstore"},
                                              {"address", address()},
                                              {"path", "large/"}},
                                             context)
                      .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, "large_value", value).result());
  EXPECT_THAT(generation.generation, IsRegularStorageGeneration());

  // memory kvstore driver returns the current time as the timestamp.
  auto now = absl::Now();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, "large_value").result());

  EXPECT_EQ(value, result.value);
  EXPECT_THAT(result.stamp, MatchesTimestampedStorageGeneration(
                                generation.generation, testing::Ge(now)));
}

TEST_F(KvStoreTest, DeleteEmptyKeyAndUnsetByteRangeExclusiveMax) {
  auto context = tensorstore::Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::kvstore::Open(
          {{"driver", "tsgrpc_kvstore"}, {"address", address()}}, context)
          .result());

  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "", absl::Cord("empty_key_val")));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_empty,
                                   kvstore::Read(store, "").result());
  EXPECT_EQ("empty_key_val", read_empty.value);

  TENSORSTORE_EXPECT_OK(kvstore::Delete(store, ""));
  EXPECT_THAT(kvstore::Read(store, "").result(),
              MatchesKvsReadResultNotFound());

  TENSORSTORE_EXPECT_OK(
      kvstore::Write(store, "suffix_test", absl::Cord("0123456789")));

  auto channel =
      grpc::CreateChannel(address(), grpc::InsecureChannelCredentials());
  auto stub =
      tensorstore_grpc::kvstore::grpc_gen::KvStoreService::NewStub(channel);
  grpc::ClientContext client_context;
  tensorstore_grpc::kvstore::ReadRequest req;
  req.set_key("suffix_test");
  req.mutable_byte_range()->set_inclusive_min(4);
  auto reader = stub->Read(&client_context, req);
  tensorstore_grpc::kvstore::ReadResponse resp;
  std::string value;
  while (reader->Read(&resp)) {
    value.append(std::string(resp.value_part()));
  }
  EXPECT_TRUE(reader->Finish().ok());
  EXPECT_EQ("456789", value);
}

}  // namespace
