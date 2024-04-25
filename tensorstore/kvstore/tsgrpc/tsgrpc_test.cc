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

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/internal/grpc/grpc_mock.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/tsgrpc/mock_kvstore_service.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/status_testutil.h"

// protos
#include "tensorstore/kvstore/tsgrpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.pb.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::KeyRange;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::ParseTextProtoOrDie;
using ::tensorstore::StorageGeneration;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

using ::tensorstore_grpc::MockKvStoreService;
using ::tensorstore_grpc::kvstore::DeleteRequest;
using ::tensorstore_grpc::kvstore::DeleteResponse;
using ::tensorstore_grpc::kvstore::ListRequest;
using ::tensorstore_grpc::kvstore::ListResponse;
using ::tensorstore_grpc::kvstore::ReadRequest;
using ::tensorstore_grpc::kvstore::ReadResponse;
using ::tensorstore_grpc::kvstore::WriteRequest;
using ::tensorstore_grpc::kvstore::WriteResponse;

class TsGrpcMockTest : public testing::Test {
 public:
  ~TsGrpcMockTest() override { mock_service_.Shutdown(); }
  TsGrpcMockTest() {
    /// Unmatched calls all return CANCELLED.
    ON_CALL(mock(), Read).WillByDefault(Return(grpc::Status::CANCELLED));
    ON_CALL(mock(), Write).WillByDefault(Return(grpc::Status::CANCELLED));
    ON_CALL(mock(), Delete).WillByDefault(Return(grpc::Status::CANCELLED));
    ON_CALL(mock(), List).WillByDefault(Return(grpc::Status::CANCELLED));
  }

  tensorstore::KvStore OpenStore() {
    return kvstore::Open({
                             {"driver", "tsgrpc_kvstore"},
                             {"address", mock_service_.server_address()},
                         })
        .value();
  }

  MockKvStoreService& mock() { return *mock_service_.service(); }

  tensorstore::grpc_mocker::MockGrpcServer<MockKvStoreService> mock_service_;
};

TEST_F(TsGrpcMockTest, Read) {
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

  EXPECT_CALL(mock(), Read(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  kvstore::ReadResult result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Read(store, expected_request.key()).result());
  }

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "1234");
  EXPECT_EQ(result.stamp.time,
            absl::FromUnixSeconds(1634327736) + absl::Nanoseconds(123456));
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, ReadWithOptions) {
  ReadRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: "abc"
    generation_if_not_equal: "abc\001"
    generation_if_equal: "xyz\001"
    byte_range { inclusive_min: 1 exclusive_max: 10 }
  )pb");

  EXPECT_CALL(mock(), Read(_, EqualsProto(expected_request), _))
      .WillOnce(Return(grpc::Status::OK));

  kvstore::ReadResult result;
  {
    kvstore::ReadOptions options;
    options.generation_conditions.if_not_equal =
        StorageGeneration::FromString("abc");
    options.generation_conditions.if_equal =
        StorageGeneration::FromString("xyz");
    options.staleness_bound = absl::InfiniteFuture();
    options.byte_range = OptionalByteRangeRequest{1, 10};

    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Read(store, expected_request.key(), options).result());
  }
  EXPECT_EQ(result.stamp.generation, StorageGeneration::Unknown());
}

TEST_F(TsGrpcMockTest, Write) {
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

  EXPECT_CALL(mock(), Write(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Write(store, expected_request.key(),
                               absl::Cord(expected_request.value()))
                    .result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, WriteEmpty) {
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

  EXPECT_CALL(mock(), Write(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Write(store, expected_request.key(), absl::Cord(),
                               {StorageGeneration::NoValue()})
                    .result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, WriteWithOptions) {
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

  EXPECT_CALL(mock(), Write(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Write(store, expected_request.key(),
                               absl::Cord(expected_request.value()),
                               {StorageGeneration::FromString("abc")})
                    .result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, WriteNullopt) {
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

  EXPECT_CALL(mock(), Delete(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Write(store, expected_request.key(), std::nullopt,
                               {StorageGeneration::NoValue()})
                    .result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, Delete) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    key: 'abc'
  )pb");

  DeleteResponse response = ParseTextProtoOrDie(R"pb(
    generation_and_timestamp {
      generation: '1\001'
      timestamp { seconds: 1634327736 nanos: 123456 }
    }
  )pb");

  EXPECT_CALL(mock(), Delete(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Delete(store, expected_request.key()).result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, DeleteWithOptions) {
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

  EXPECT_CALL(mock(), Delete(_, EqualsProto(expected_request), _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  tensorstore::TimestampedStorageGeneration result;
  {
    auto store = OpenStore();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        result, kvstore::Delete(store, expected_request.key(),
                                {StorageGeneration::FromString("abc")})
                    .result());
  }
  EXPECT_EQ(result.generation, StorageGeneration::FromString("1"));
}

TEST_F(TsGrpcMockTest, DeleteRange) {
  DeleteRequest expected_request = ParseTextProtoOrDie(R"pb(
    range { inclusive_min: 'a/c' exclusive_max: 'a/d' }
  )pb");

  EXPECT_CALL(mock(), Delete(_, EqualsProto(expected_request), _))
      .WillOnce(Return(grpc::Status::OK));

  {
    auto store = OpenStore();
    TENSORSTORE_EXPECT_OK(
        kvstore::DeleteRange(store, KeyRange::Prefix("a/c")).result());
  }
}

// List is special; it doesn't use the async() callback interface because
// it needs to run in a thread for each of the tensorstore::execution calls.
TEST_F(TsGrpcMockTest, List) {
  ListRequest expected_request = ParseTextProtoOrDie(R"pb(
    range: {}
  )pb");

  ListResponse response = ParseTextProtoOrDie(R"pb(
    entry { key: 'a' }
    entry { key: 'b' }
    entry { key: 'c' }
  )pb");

  // Set expectation and action on the mock stub.
  EXPECT_CALL(mock(), List(_, EqualsProto(expected_request), _))
      .WillOnce(testing::Invoke(
          [=](auto*, auto*,
              grpc::ServerWriter<ListResponse>* resp) -> ::grpc::Status {
            resp->Write(response);
            return grpc::Status::OK;
          }));

  std::vector<std::string> log;
  {
    auto store = OpenStore();

    // Listing the entire stream works.
    absl::Notification notification;
    tensorstore::execution::submit(
        tensorstore::kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});

    notification.WaitForNotification();
  }

  EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                       "set_starting", "set_value: a", "set_value: b",
                       "set_value: c", "set_done", "set_stopping"));
}

}  // namespace
