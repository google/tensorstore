// Copyright 2023 The TensorStore Authors
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

#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/grpc/grpc_mock.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/kvstore/gcs_grpc/mock_storage_service.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

// protos
#include "google/storage/v2/storage.pb.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::KeyRange;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ParseTextProtoOrDie;
using ::tensorstore::StorageGeneration;
using ::tensorstore::grpc_mocker::MockGrpcServer;
using ::tensorstore::internal::AbslStatusToGrpcStatus;
using ::tensorstore_grpc::MockStorage;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

using ::google::storage::v2::DeleteObjectRequest;
using ::google::storage::v2::ListObjectsRequest;
using ::google::storage::v2::ListObjectsResponse;
using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::ReadObjectResponse;
using ::google::storage::v2::WriteObjectRequest;
using ::google::storage::v2::WriteObjectResponse;

class GcsGrpcTest : public testing::Test {
 public:
  tensorstore::KvStore OpenStore() {
    ABSL_LOG(INFO) << "Using " << mock_service_.server_address();
    return kvstore::Open({{"driver", "gcs_grpc"},
                          {"endpoint", mock_service_.server_address()},
                          {"bucket", "bucket"},
                          {"timeout", "100ms"}})
        .value();
  }

  MockStorage& mock() { return *mock_service_.service(); }

  tensorstore::grpc_mocker::MockGrpcServer<MockStorage> mock_service_;
};

TEST_F(GcsGrpcTest, Read) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  ReadObjectResponse response = ParseTextProtoOrDie(R"pb(
    metadata { generation: 2 }
    checksummed_data { content: '1234' }
  )pb");

  // Set expectation and action on the mock stub.
  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .WillOnce(testing::Invoke(
          [&](auto*, auto*,
              grpc::ServerWriter<ReadObjectResponse>* resp) -> ::grpc::Status {
            resp->Write(response);
            return grpc::Status::OK;
          }));

  auto start = absl::Now();
  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, expected_request.object()).result());

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "1234");
  EXPECT_GT(result.stamp.time, start);
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, ReadRetry) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  ReadObjectResponse response = ParseTextProtoOrDie(R"pb(
    metadata { generation: 2 }
    checksummed_data { content: '1234' }
  )pb");

  // Set expectation and action on the mock stub.
  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))))
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))))
      .WillOnce(testing::Invoke(
          [&](auto*, auto*,
              grpc::ServerWriter<ReadObjectResponse>* resp) -> ::grpc::Status {
            resp->Write(response);
            return grpc::Status::OK;
          }));

  auto start = absl::Now();
  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, expected_request.object()).result());

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "1234");
  EXPECT_GT(result.stamp.time, start);
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, ReadWithOptions) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    if_generation_not_match: 3
    if_generation_match: 1
    read_offset: 1
    read_limit: 9
  )pb");

  ReadObjectResponse response = ParseTextProtoOrDie(R"pb(
    metadata { generation: 2 }
    checksummed_data { content: '1234' }
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .WillOnce(testing::Invoke(
          [&](auto*, auto*,
              grpc::ServerWriter<ReadObjectResponse>* resp) -> ::grpc::Status {
            resp->Write(response);
            return grpc::Status::OK;
          }));

  kvstore::ReadOptions options;
  options.if_not_equal = StorageGeneration::FromUint64(3);
  options.if_equal = StorageGeneration::FromUint64(1);
  options.staleness_bound = absl::InfiniteFuture();
  options.byte_range = {1, 10};

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Read(store, expected_request.object(), options).result());
}

TEST_F(GcsGrpcTest, Write) {
  std::vector<WriteObjectRequest> requests;

  WriteObjectResponse response = ParseTextProtoOrDie(R"pb(
    resource { name: 'abc' bucket: 'bucket' generation: 1 }
  )pb");

  EXPECT_CALL(mock(), WriteObject)
      .WillOnce(testing::Invoke(
          [&](auto*, grpc::ServerReader<WriteObjectRequest>* reader,
              auto* resp) -> ::grpc::Status {
            WriteObjectRequest req;
            while (reader->Read(&req)) {
              requests.push_back(req);
            }
            resp->CopyFrom(response);
            return grpc::Status::OK;
          }));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, response.resource().name(), absl::Cord("abcd"))
          .result());

  EXPECT_THAT(requests,
              testing::ElementsAre(EqualsProto<WriteObjectRequest>(R"pb(
                write_object_spec {
                  resource { name: "abc" bucket: "projects/_/buckets/bucket" }
                  object_size: 4
                }
                checksummed_data { content: "abcd" crc32c: 2462583345 }
                object_checksums { crc32c: 2462583345 }
                finish_write: true
                write_offset: 0
              )pb")));
}

TEST_F(GcsGrpcTest, WriteRetry) {
  std::vector<WriteObjectRequest> requests;

  WriteObjectResponse response = ParseTextProtoOrDie(R"pb(
    resource { name: 'abc' bucket: 'bucket' generation: 1 }
  )pb");

  EXPECT_CALL(mock(), WriteObject)
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))))
      .WillOnce(testing::Invoke(
          [&](auto*, grpc::ServerReader<WriteObjectRequest>* reader,
              auto* resp) -> ::grpc::Status {
            WriteObjectRequest req;
            if (reader->Read(&req)) {
              requests.push_back(req);
            }
            return AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""));
          }))
      .WillOnce(testing::Invoke(
          [&](auto*, grpc::ServerReader<WriteObjectRequest>* reader,
              auto* resp) -> ::grpc::Status {
            WriteObjectRequest req;
            while (reader->Read(&req)) {
              requests.push_back(req);
            }
            resp->CopyFrom(response);
            return grpc::Status::OK;
          }));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, response.resource().name(), absl::Cord("abcd"))
          .result());

  EXPECT_THAT(
      requests,
      testing::ElementsAre(
          EqualsProto<WriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "abc" bucket: "projects/_/buckets/bucket" }
              object_size: 4
            }
            checksummed_data { content: "abcd" crc32c: 2462583345 }
            object_checksums { crc32c: 2462583345 }
            finish_write: true
            write_offset: 0
          )pb"),
          EqualsProto<WriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "abc" bucket: "projects/_/buckets/bucket" }
              object_size: 4
            }
            checksummed_data { content: "abcd" crc32c: 2462583345 }
            object_checksums { crc32c: 2462583345 }
            finish_write: true
            write_offset: 0
          )pb")));
}

TEST_F(GcsGrpcTest, WriteEmpty) {
  std::vector<WriteObjectRequest> requests;

  WriteObjectResponse response = ParseTextProtoOrDie(R"pb(
    resource { name: 'abc' bucket: 'projects/_/buckets/bucket' generation: 1 }
  )pb");

  EXPECT_CALL(mock(), WriteObject)
      .WillOnce(testing::Invoke(
          [&](auto*, grpc::ServerReader<WriteObjectRequest>* reader,
              auto* resp) -> ::grpc::Status {
            WriteObjectRequest req;
            while (reader->Read(&req)) {
              requests.push_back(req);
            }
            resp->CopyFrom(response);
            return grpc::Status::OK;
          }));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, response.resource().name(), absl::Cord()).result());

  EXPECT_THAT(requests,
              testing::ElementsAre(EqualsProto<WriteObjectRequest>(R"pb(
                write_object_spec {
                  resource { name: "abc" bucket: "projects/_/buckets/bucket" }
                  object_size: 0
                }
                checksummed_data { crc32c: 0 }
                object_checksums { crc32c: 0 }
                finish_write: true
                write_offset: 0
              )pb")));
}

TEST_F(GcsGrpcTest, WriteWithOptions) {
  std::vector<WriteObjectRequest> requests;

  WriteObjectResponse response = ParseTextProtoOrDie(R"pb(
    resource { name: 'abc' bucket: 'bucket' generation: 1 }
  )pb");

  EXPECT_CALL(mock(), WriteObject)
      .WillOnce(testing::Invoke(
          [&](auto*, grpc::ServerReader<WriteObjectRequest>* reader,
              auto* resp) -> ::grpc::Status {
            WriteObjectRequest req;
            while (reader->Read(&req)) {
              requests.push_back(req);
            }
            resp->CopyFrom(response);
            return grpc::Status::OK;
          }));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, response.resource().name(), absl::Cord("abcd"),
                     {StorageGeneration::FromUint64(3)})
          .result());

  EXPECT_THAT(requests,
              testing::ElementsAre(EqualsProto<WriteObjectRequest>(R"pb(
                write_object_spec {
                  resource { name: "abc" bucket: "projects/_/buckets/bucket" }
                  if_generation_match: 3
                  object_size: 4
                }
                checksummed_data { content: "abcd" crc32c: 2462583345 }
                object_checksums { crc32c: 2462583345 }
                finish_write: true
                write_offset: 0
              )pb")));
}

TEST_F(GcsGrpcTest, WriteNullopt) {
  DeleteObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    if_generation_match: 0
  )pb");

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(expected_request), _))
      .WillOnce(Return(grpc::Status::OK));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, expected_request.object(), std::nullopt,
                     {StorageGeneration::NoValue()})
          .result());
}

TEST_F(GcsGrpcTest, Delete) {
  DeleteObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(expected_request), _))
      .WillOnce(Return(grpc::Status::OK));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Delete(store, expected_request.object()).result());
}

TEST_F(GcsGrpcTest, DeleteWithOptions) {
  DeleteObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    if_generation_match: 2
  )pb");

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(expected_request), _))
      .WillOnce(Return(grpc::Status::OK));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Delete(store, expected_request.object(),
                                       {StorageGeneration::FromUint64(2)})
                           .result());
}

TEST_F(GcsGrpcTest, DeleteRange) {
  ListObjectsRequest request1 = ParseTextProtoOrDie(R"pb(
    parent: 'projects/_/buckets/bucket'
    page_size: 1000
    lexicographic_start: 'a/c'
    lexicographic_end: 'a/d'
  )pb");

  ListObjectsResponse response1 = ParseTextProtoOrDie(R"pb(
    objects { name: 'a/c' }
    objects { name: 'a/ce' }
  )pb");

  DeleteObjectRequest request2 = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'a/c'
  )pb");

  DeleteObjectRequest request3 = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'a/ce'
  )pb");

  EXPECT_CALL(mock(), ListObjects(_, EqualsProto(request1), _))
      .WillOnce(DoAll(SetArgPointee<2>(response1), Return(grpc::Status::OK)));

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(request2), _))
      .WillOnce(Return(grpc::Status::OK));

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(request3), _))
      .WillOnce(Return(grpc::Status::OK));

  auto store = OpenStore();
  TENSORSTORE_EXPECT_OK(
      kvstore::DeleteRange(store, KeyRange::Prefix("a/c")).result());
}

// List is special; it doesn't use the async() callback interface because
// it needs to run in a thread for each of the tensorstore::execution calls.
TEST_F(GcsGrpcTest, List) {
  ListObjectsRequest request1 = ParseTextProtoOrDie(R"pb(
    parent: 'projects/_/buckets/bucket'
    page_size: 1000
  )pb");

  ListObjectsRequest request2 = ParseTextProtoOrDie(R"pb(
    parent: 'projects/_/buckets/bucket'
    page_size: 1000
    page_token: 'next-page-token'
  )pb");

  ListObjectsResponse response1 = ParseTextProtoOrDie(R"pb(
    objects { name: 'a' }
    objects { name: 'b' }
    next_page_token: 'next-page-token'
  )pb");

  ListObjectsResponse response2 = ParseTextProtoOrDie(R"pb(
    objects { name: 'c' }
  )pb");

  EXPECT_CALL(mock(), ListObjects(_, EqualsProto(request1), _))
      .WillOnce(DoAll(SetArgPointee<2>(response1), Return(grpc::Status::OK)));
  EXPECT_CALL(mock(), ListObjects(_, EqualsProto(request2), _))
      .WillOnce(DoAll(SetArgPointee<2>(response2), Return(grpc::Status::OK)));

  // Listing the entire stream works.
  auto store = OpenStore();

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

TEST(GcsGrpcSpecTest, InvalidSpec) {
  auto context = Context::Default();

  // Test with missing `"bucket"` key.
  EXPECT_THAT(kvstore::Open({{"driver", "gcs_grpc"}}, context).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"bucket"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "gcs_grpc"}, {"bucket", "bucket:xyz"}}, context)
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(GcsGrpcUrlTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "gcs_grpc"}, {"bucket", "my-bucket"}, {"path", "abc"}},
      "gcs_grpc://my-bucket/abc");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "gcs_grpc"}, {"bucket", "my-bucket"}, {"path", "abc def"}},
      "gcs_grpc://my-bucket/abc%20def");
}

TEST(GcsGrpcUrlTest, InvalidUri) {
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket:xyz"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Invalid bucket name: \"bucket:xyz\""));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket?query"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Query string not supported"));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket#fragment"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*: Fragment identifier not supported"));
}

}  // namespace
