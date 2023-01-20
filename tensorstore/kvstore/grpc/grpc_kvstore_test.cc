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

#include "tensorstore/kvstore/grpc/grpc_kvstore.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/completion_queue.h"  // third_party
#include "grpcpp/support/async_stream.h"  // third_party
#include "grpcpp/support/async_unary_call.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "grpcpp/test/mock_stream.h"  // third_party
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

// protos

#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.pb.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::KeyRange;
using ::tensorstore::ParseTextProtoOrDie;
using ::tensorstore::StorageGeneration;
using ::tensorstore_grpc::kvstore::DeleteRequest;
using ::tensorstore_grpc::kvstore::DeleteResponse;
using ::tensorstore_grpc::kvstore::ListRequest;
using ::tensorstore_grpc::kvstore::ListResponse;
using ::tensorstore_grpc::kvstore::ReadRequest;
using ::tensorstore_grpc::kvstore::ReadResponse;
using ::tensorstore_grpc::kvstore::WriteRequest;
using ::tensorstore_grpc::kvstore::WriteResponse;
using ::tensorstore_grpc::kvstore::grpc_gen::KvStoreService;
using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::Return;
using ::testing::SetArgPointee;

class MockKvStoreServiceStub : public KvStoreService::StubInterface {
 public:
  MOCK_METHOD(::grpc::Status, Read,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ReadRequest& request,
               ::tensorstore_grpc::kvstore::ReadResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::ReadResponse>*,
              AsyncReadRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ReadRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::ReadResponse>*,
              PrepareAsyncReadRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ReadRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::Status, Write,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::WriteRequest& request,
               ::tensorstore_grpc::kvstore::WriteResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::WriteResponse>*,
              AsyncWriteRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::WriteRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::WriteResponse>*,
              PrepareAsyncWriteRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::WriteRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::Status, Delete,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::DeleteRequest& request,
               ::tensorstore_grpc::kvstore::DeleteResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::DeleteResponse>*,
              AsyncDeleteRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::DeleteRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::tensorstore_grpc::kvstore::DeleteResponse>*,
              PrepareAsyncDeleteRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::DeleteRequest& request,
               ::grpc::CompletionQueue* cq));
  MOCK_METHOD(::grpc::ClientReaderInterface<
                  ::tensorstore_grpc::kvstore::ListResponse>*,
              ListRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ListRequest& request));
  MOCK_METHOD(::grpc::ClientAsyncReaderInterface<
                  ::tensorstore_grpc::kvstore::ListResponse>*,
              AsyncListRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ListRequest& request,
               ::grpc::CompletionQueue* cq, void* tag));
  MOCK_METHOD(::grpc::ClientAsyncReaderInterface<
                  ::tensorstore_grpc::kvstore::ListResponse>*,
              PrepareAsyncListRaw,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ListRequest& request,
               ::grpc::CompletionQueue* cq));

  MOCK_METHOD(async_interface*, async, ());
};

class MockKvStoreServiceStubAsyncInterface
    : public KvStoreService::StubInterface::async_interface {
 public:
  MOCK_METHOD(void, Read,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ReadRequest* request,
               ::tensorstore_grpc::kvstore::ReadResponse* response,
               std::function<void(::grpc::Status)> fn));
  MOCK_METHOD(void, Read,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::ReadRequest* request,
               ::tensorstore_grpc::kvstore::ReadResponse* response,
               ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(void, Write,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::WriteRequest* request,
               ::tensorstore_grpc::kvstore::WriteResponse* response,
               std::function<void(::grpc::Status)> fn));
  MOCK_METHOD(void, Write,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::WriteRequest* request,
               ::tensorstore_grpc::kvstore::WriteResponse* response,
               ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(void, Delete,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::DeleteRequest* request,
               ::tensorstore_grpc::kvstore::DeleteResponse* response,
               std::function<void(::grpc::Status)> fn));
  MOCK_METHOD(void, Delete,
              (::grpc::ClientContext * context,
               const ::tensorstore_grpc::kvstore::DeleteRequest* request,
               ::tensorstore_grpc::kvstore::DeleteResponse* response,
               ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(
      void, List,
      (::grpc::ClientContext * context,
       const ::tensorstore_grpc::kvstore::ListRequest* request,
       ::grpc::ClientReadReactor< ::tensorstore_grpc::kvstore::ListResponse>*
           reactor));
};

class KvStoreMockTest : public testing::Test {
 public:
  using Fn = std::function<void(::grpc::Status)>;

  KvStoreMockTest() { ON_CALL(*mock_, async()).WillByDefault(Return(&async_)); }

  kvstore::DriverPtr OpenStore() {
    auto store_result =
        tensorstore::CreateGrpcKvStore("[::1]:0", absl::Seconds(2), mock_);

    EXPECT_TRUE(store_result.ok()) << store_result.status();
    return std::move(store_result).value();
  }

  std::shared_ptr<MockKvStoreServiceStub> mock_ =
      std::make_shared<MockKvStoreServiceStub>();

  MockKvStoreServiceStubAsyncInterface async_;
};

TEST_F(KvStoreMockTest, Read) {
  ReadRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
  )pb");

  ReadResponse response = ParseTextProtoOrDie(R"pb(
    state: 2
    value: '1234'
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Read(_, EqualsProto(expected_request), _,
                           testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, expected_request.key()).result());

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "1234");
  EXPECT_EQ(result.stamp.time,
            absl::FromUnixSeconds(1634327736) + absl::Nanoseconds(123456));
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromString("1"));
}

TEST_F(KvStoreMockTest, ReadWithOptions) {
  ReadRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: "abc"
    generation_if_not_equal: "abc\001"
    generation_if_equal: "xyz\001"
    byte_range { inclusive_min: 1 exclusive_max: 10 }
  )pb");

  ReadResponse response;

  EXPECT_CALL(async_, Read(_, EqualsProto(expected_request), _,
                           testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  kvstore::ReadOptions options;
  options.if_not_equal = StorageGeneration::FromString("abc");
  options.if_equal = StorageGeneration::FromString("xyz");
  options.staleness_bound = absl::InfiniteFuture();
  options.byte_range = {1, 10};

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Read(store, expected_request.key(), options).result());
}

TEST_F(KvStoreMockTest, Write) {
  WriteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
    value: '1234'
  )pb");

  WriteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Write(_, EqualsProto(expected_request), _,
                            testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, expected_request.key(),
                                      absl::Cord(expected_request.value()))
                           .result());
}

TEST_F(KvStoreMockTest, WriteEmpty) {
  WriteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
    generation_if_equal: '\005'
  )pb");

  WriteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Write(_, EqualsProto(expected_request), _,
                            testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, expected_request.key(),
                                      absl::Cord(expected_request.value()),
                                      {StorageGeneration::NoValue()})
                           .result());
}

TEST_F(KvStoreMockTest, WriteWithOptions) {
  WriteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
    value: '1234'
    generation_if_equal: "abc\001"
  )pb");

  WriteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Write(_, EqualsProto(expected_request), _,
                            testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, expected_request.key(),
                                      absl::Cord(expected_request.value()),
                                      {StorageGeneration::FromString("abc")})
                           .result());
}

TEST_F(KvStoreMockTest, WriteNullopt) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
    generation_if_equal: '\005'
  )pb");

  DeleteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Delete(_, EqualsProto(expected_request), _,
                             testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, expected_request.key(), std::nullopt,
                     {StorageGeneration::NoValue()})
          .result());
}

TEST_F(KvStoreMockTest, Delete) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
  )pb");

  DeleteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Delete(_, EqualsProto(expected_request), _,
                             testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Delete(store, expected_request.key()).result());
}

TEST_F(KvStoreMockTest, DeleteWithOptions) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
    generation_if_equal: "abc\001"
  )pb");

  DeleteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(async_, Delete(_, EqualsProto(expected_request), _,
                             testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Delete(store, expected_request.key(),
                                       {StorageGeneration::FromString("abc")})
                           .result());
}

TEST_F(KvStoreMockTest, DeleteRange) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    range { inclusive_min: 'a/c' exclusive_max: 'a/d' }
  )pb");

  DeleteResponse response;

  EXPECT_CALL(async_, Delete(_, EqualsProto(expected_request), _,
                             testing::Matcher<Fn>(_)))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      InvokeArgument<3>(grpc::Status::OK)));

  auto store = OpenStore();
  TENSORSTORE_EXPECT_OK(
      kvstore::DeleteRange(store, KeyRange::Prefix("a/c")).result());
}

// List is special; it doesn't use the async() callback interface because
// it needs to run in a thread for each of the tensorstore::execution calls.
TEST_F(KvStoreMockTest, List) {
  auto store = OpenStore();

  ListRequest expected_request = ParseTextProtoOrDie(R"pb(
    range: {}
  )pb");
  ListResponse response = ParseTextProtoOrDie(R"pb(
    key: 'a' key: 'b' key: 'c'
  )pb");

  // The ownership is transferred to the client.
  {
    auto* mock_reader = new ::grpc::testing::MockClientReader<ListResponse>();

    // Set expectation and action on the mock stub.
    EXPECT_CALL(*mock_, ListRaw(_, EqualsProto(expected_request)))
        .WillOnce(Return(mock_reader));
    // Set expectation and action on the mock reader object.
    EXPECT_CALL(*mock_reader, Read(_))
        .WillOnce(DoAll(SetArgPointee<0>(response), Return(true)))
        .WillOnce(Return(false));
    EXPECT_CALL(*mock_reader, Finish()).WillOnce(Return(::grpc::Status::OK));
  }

  // Listing the entire stream works.
  absl::Notification notification;
  std::vector<std::string> log;
  tensorstore::execution::submit(
      tensorstore::kvstore::List(store, {}),
      tensorstore::CompletionNotifyingReceiver{
          &notification, tensorstore::LoggingReceiver{&log}});
  notification.WaitForNotification();

  EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                       "set_starting", "set_value: a", "set_value: b",
                       "set_value: c", "set_done", "set_stopping"));
}

}  // namespace
