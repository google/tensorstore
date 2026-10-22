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

#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/grpc/grpc_mock.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/gcs/validate.h"
#include "tensorstore/kvstore/gcs_grpc/default_strategy.h"
#include "tensorstore/kvstore/gcs_grpc/mock_storage_service.h"
#include "tensorstore/kvstore/gcs_grpc/op_bidi_write.h"
#include "tensorstore/kvstore/gcs_grpc/op_move.h"
#include "tensorstore/kvstore/gcs_grpc/test_hook.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

// protos
#include "google/storage/v2/storage.pb.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::Future;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::KeyRange;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesJson;
using ::tensorstore::OptionalByteRangeRequest;
using ::tensorstore::ParseTextProtoOrDie;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::grpc_mocker::MockGrpcServer;
using ::tensorstore::internal::FlatCordBuilder;
using ::tensorstore::internal_gcs_grpc::IsAppendableNotSupportedError;
using ::tensorstore::internal_gcs_grpc::IsZonalObjectRequiredError;
using ::tensorstore::internal_gcs_grpc::TestSetIsZonal;
using ::tensorstore::kvstore::SupportedFeatures;
using ::tensorstore_grpc::MatchesTempWriteRequest;
using ::tensorstore_grpc::MockStorage;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::SetArgPointee;

using ::google::storage::v2::BidiWriteObjectRequest;
using ::google::storage::v2::BidiWriteObjectResponse;
using ::google::storage::v2::DeleteObjectRequest;
using ::google::storage::v2::ListObjectsRequest;
using ::google::storage::v2::ListObjectsResponse;
using ::google::storage::v2::MoveObjectRequest;
using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::ReadObjectResponse;

// NOTE: The mock mechanism should not timeout unexpectedly, yet it does.
// Investigate these cases by replacing AtLeast with Exactly.

class GcsGrpcTest : public testing::Test {
 public:
  void SetUp() override {
    ON_CALL(mock(), MoveObject(_, _, _))
        .WillByDefault([](auto*, const MoveObjectRequest* req, auto* resp) {
          resp->set_name(req->destination_object());
          resp->set_bucket(req->bucket());
          resp->set_generation(req->has_if_source_generation_match()
                                   ? req->if_source_generation_match()
                                   : 42);
          return grpc::Status::OK;
        });
  }

  tensorstore::KvStore OpenStore() {
    ABSL_LOG(INFO) << "Using " << mock_service_.server_address();
    return kvstore::Open({{"driver", "gcs_grpc"},
                          {"endpoint", mock_service_.server_address()},
                          {"bucket", "bucket"},
                          {"timeout", "250ms"}})
        .value();
  }

  // Opens a store with an explicit `mode` setting (`true` -> `"zonal"`,
  // `false` -> `"standard"`).  Defaults to `"zonal"`, which is what the
  // atomic-write tests exercise.
  tensorstore::KvStore OpenRapidStore(std::optional<bool> rapid = true) {
    if (!rapid.has_value()) return OpenStore();
    return OpenModeStore(*rapid ? "zonal" : "standard");
  }

  tensorstore::KvStore OpenModeStore(std::string_view mode) {
    ABSL_LOG(INFO) << "Using " << mock_service_.server_address();
    return kvstore::Open({{"driver", "gcs_grpc"},
                          {"endpoint", mock_service_.server_address()},
                          {"bucket", "bucket"},
                          {"timeout", "250ms"},
                          {"experimental_mode", mode}})
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

  // Set expectation and action on the mock stub.
  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: '1234' }
            )pb"));
            return grpc::Status::OK;
          });

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

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: '1234' }
            )pb"));
            return grpc::Status::OK;
          });

  kvstore::ReadOptions options;
  options.generation_conditions.if_not_equal = StorageGeneration::FromUint64(3);
  options.generation_conditions.if_equal = StorageGeneration::FromUint64(1);
  options.staleness_bound = absl::InfiniteFuture();
  options.byte_range = OptionalByteRangeRequest{1, 10};

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, "abc", options).result());
}

TEST_F(GcsGrpcTest, ReadStatCrc32c) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    read_limit: 1
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            ReadObjectResponse response = ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              object_checksums { crc32c: 12345678 }
            )pb");
            resp->Write(response);
            return grpc::Status::OK;
          });

  kvstore::ReadOptions options;
  options.byte_range = OptionalByteRangeRequest::Stat();

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, "abc", options).result());

  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value.size(), 0);
}

TEST_F(GcsGrpcTest, ReadSingleByteRangeCrc32c) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    read_offset: 10
    read_limit: 1
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            ReadObjectResponse response = ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              object_checksums { crc32c: 12345678 }
              checksummed_data { content: 'X' }
            )pb");
            resp->Write(response);
            return grpc::Status::OK;
          });

  kvstore::ReadOptions options;
  options.byte_range = OptionalByteRangeRequest{10, 11};

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Read(store, "abc", options).result());

  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "X");
}

TEST_F(GcsGrpcTest, ReadMultipleMessages) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              checksummed_data { content: '123' }
            )pb"));
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: '456' }
            )pb"));
            return grpc::Status::OK;
          });

  auto start = absl::Now();
  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result,
                                   kvstore::Read(store, "abc").result());

  // Individual result field verification.
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "123456");
  EXPECT_GT(result.stamp.time, start);
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, ReadInBatchAdjacent) {
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    read_offset: 0
    read_limit: 20
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: '0123456789abcdefghij' }
            )pb"));
            return grpc::Status::OK;
          });

  auto store = OpenStore();
  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 11;
    options.byte_range.exclusive_max = 20;
    read2 = kvstore::Read(store, "abc", options);

    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "0123456789");
  EXPECT_EQ(result1.stamp.generation, StorageGeneration::FromUint64(2));

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "bcdefghij");
  EXPECT_EQ(result2.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, ReadInBatchUnboundedOverlapping) {
  // Only range requests are coalesced by the generic batch read handler.
  std::string_view value = "0123456789abcdefghij";

  EXPECT_CALL(mock(), ReadObject(_, _, _))
      .Times(AtLeast(1))
      .WillRepeatedly([value](auto* context, const ReadObjectRequest* request,
                              grpc::ServerWriter<ReadObjectResponse>* resp)
                          -> grpc::Status {
        ABSL_LOG(INFO) << "ReadObject: " << request->ShortDebugString();
        EXPECT_THAT(*request,
                    testing::AnyOf(EqualsProto(
                                       R"pb(bucket: "projects/_/buckets/bucket"
                                            object: "abc"
                                            read_offset: 5
                                       )pb"),
                                   EqualsProto(
                                       R"pb(bucket: "projects/_/buckets/bucket"
                                            object: "abc"
                                            read_offset: 0
                                            read_limit: 10
                                       )pb")));
        int n = request->read_limit()
                    ? request->read_limit() - request->read_offset()
                    : value.size() - request->read_offset();
        ReadObjectResponse response;
        response.mutable_metadata()->set_generation(2);
        response.mutable_checksummed_data()->set_content(
            value.substr(request->read_offset(), n));
        resp->Write(response);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;
    options.byte_range.inclusive_min = 5;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "56789abcdefghij");
  EXPECT_EQ(result1.stamp.generation, StorageGeneration::FromUint64(2));

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "0123456789");
  EXPECT_EQ(result2.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, ReadInBatchNonAdjacent) {
  ReadObjectRequest request1 = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    read_offset: 0
    read_limit: 10
  )pb");

  ReadObjectRequest request2 = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    read_offset: 100010
    read_limit: 10
  )pb");

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(request1), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: '0123456789' }
            )pb"));
            return grpc::Status::OK;
          });

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(request2), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 }
              checksummed_data { content: 'abcdefghij' }
            )pb"));
            return grpc::Status::OK;
          });

  auto store = OpenStore();
  Future<kvstore::ReadResult> read1;
  Future<kvstore::ReadResult> read2;
  {
    auto batch = tensorstore::Batch::New();

    kvstore::ReadOptions options;
    options.batch = batch;

    options.byte_range.inclusive_min = 0;
    options.byte_range.exclusive_max = 10;
    read1 = kvstore::Read(store, "abc", options);

    options.byte_range.inclusive_min = 100010;
    options.byte_range.exclusive_max = 100020;
    read2 = kvstore::Read(store, "abc", options);
    batch.Release();
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result1, read1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result2, read2.result());

  // Individual result field verification.
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value, "0123456789");
  EXPECT_EQ(result1.stamp.generation, StorageGeneration::FromUint64(2));

  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value, "abcdefghij");
  EXPECT_EQ(result2.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsGrpcTest, Write) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        while (true) {
          BidiWriteObjectRequest req;
          if (!stream->Read(&req)) break;
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          requests.push_back(std::move(req));
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("abc");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(1);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, "abc", absl::Cord("abcd")).result());

  EXPECT_THAT(
      requests,
      testing::ElementsAre(
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "abc" bucket: "projects/_/buckets/bucket" }
              object_size: 4
              appendable: false
            }
            state_lookup: true
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "abcd" crc32c: 2462583345 }
            object_checksums { crc32c: 2462583345 }
            finish_write: true
            write_offset: 0
          )pb")));
}

TEST_F(GcsGrpcTest, WriteEmpty) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject)
      .Times(AtLeast(1))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        while (true) {
          BidiWriteObjectRequest req;
          if (!stream->Read(&req)) break;
          requests.push_back(std::move(req));
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("abc");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(1);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, "abc", absl::Cord()).result());

  EXPECT_THAT(
      requests,
      testing::AllOf(
          testing::SizeIs(testing::Ge(1)),
          testing::Each(EqualsProto<BidiWriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "abc" bucket: "projects/_/buckets/bucket" }
              object_size: 0
              appendable: false
            }
            checksummed_data { crc32c: 0 }
            object_checksums { crc32c: 0 }
            finish_write: true
            write_offset: 0
          )pb"))));
}

TEST_F(GcsGrpcTest, WriteWithOptions) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject)
      .Times(AtLeast(1))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        while (true) {
          BidiWriteObjectRequest req;
          if (!stream->Read(&req)) break;
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          requests.push_back(std::move(req));
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("abc");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(1);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, "abc", absl::Cord("abcd"),
                                      {StorageGeneration::FromUint64(3)})
                           .result());

  EXPECT_THAT(
      requests,
      testing::ElementsAre(
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "abc" bucket: "projects/_/buckets/bucket" }
              if_generation_match: 3
              object_size: 4
              appendable: false
            }
            state_lookup: true
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "abcd" crc32c: 2462583345 }
            object_checksums { crc32c: 2462583345 }
            finish_write: true
            write_offset: 0
          )pb")));
}

TEST_F(GcsGrpcTest, WriteMultipleRequests) {
  std::vector<BidiWriteObjectRequest> requests;
  EXPECT_CALL(mock(), BidiWriteObject)
      .Times(AtLeast(1))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        while (true) {
          BidiWriteObjectRequest req;
          if (!stream->Read(&req)) break;
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          } else {
            size_t len = req.checksummed_data().content().size();
            req.mutable_checksummed_data()->set_content(
                absl::StrFormat("size: %d", len));
          }
          requests.push_back(std::move(req));
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("bigly");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(1);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  FlatCordBuilder cord_builder(16 + 2 * 1048576);
  memset(cord_builder.data(), 0x37, cord_builder.size());
  absl::Cord data = std::move(cord_builder).Build();
  data.Append("abcd");  // second chunk.

  EXPECT_EQ(data.size(), 2097172);

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation, kvstore::Write(store, "bigly", data).result());

  ASSERT_THAT(requests, testing::SizeIs(testing::Ge(3)));

  EXPECT_THAT(
      tensorstore::span(&requests[requests.size() - 3], 3),
      testing::ElementsAre(
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            write_object_spec {
              resource { name: "bigly" bucket: "projects/_/buckets/bucket" }
              object_size: 2097172
              appendable: false
            }
            state_lookup: true
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "size: 2097152", crc32c: 2470751355 }
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "size: 20", crc32c: 2394860217 }
            object_checksums { crc32c: 1181131586 }
            finish_write: true
            write_offset: 2097152
          )pb")));
}

TEST_F(GcsGrpcTest, WriteNullopt) {
  DeleteObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
    if_generation_match: 0
  )pb");

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(grpc::Status::OK));

  auto store = OpenStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto generation,
      kvstore::Write(store, expected_request.object(), std::nullopt,
                     {StorageGeneration::NoValue()})
          .result());
}

TEST_F(GcsGrpcTest, RapidWriteExplicit) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(
                  requests.back().write_offset() +
                  requests.back().checksummed_data().content().size());
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name(
                    requests.front().write_object_spec().resource().name());
                res->set_bucket(
                    requests.front().write_object_spec().resource().bucket());
                res->set_generation(42);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord("abcd")).result());

  EXPECT_THAT(requests,
              testing::ElementsAre(
                  MatchesTempWriteRequest("abc.__tmp_", R"pb(
                    write_object_spec {
                      resource { bucket: "projects/_/buckets/bucket" }
                      object_size: 4
                      appendable: true
                    }
                    state_lookup: true
                    write_offset: 0
                  )pb"),
                  EqualsProto<BidiWriteObjectRequest>(R"pb(
                    checksummed_data { content: "abcd" crc32c: 2462583345 }
                    object_checksums { crc32c: 2462583345 }
                    finish_write: true
                    write_offset: 0
                  )pb")));
}

TEST_F(GcsGrpcTest, RapidWriteChunked) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              size_t len = req.checksummed_data().content().size();
              if (!req.state_lookup() || req.finish_write()) {
                req.mutable_checksummed_data()->set_content(
                    absl::StrFormat("size: %d", len));
              }
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(requests.back().write_offset() + len);
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name("bigly");
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(99);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  FlatCordBuilder cord_builder(16 + 2 * 1048576);
  memset(cord_builder.data(), 0x37, cord_builder.size());
  absl::Cord data = std::move(cord_builder).Build();
  data.Append("abcd");

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "bigly", data).result());

  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
  EXPECT_THAT(
      requests,
      testing::ElementsAre(
          MatchesTempWriteRequest("bigly.__tmp_", R"pb(
            write_object_spec {
              resource { bucket: "projects/_/buckets/bucket" }
              object_size: 2097172
              appendable: true
            }
            state_lookup: true
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "size: 2097152" crc32c: 2470751355 }
            write_offset: 0
          )pb"),
          EqualsProto<BidiWriteObjectRequest>(R"pb(
            checksummed_data { content: "size: 20" crc32c: 2394860217 }
            object_checksums { crc32c: 1181131586 }
            finish_write: true
            write_offset: 2097152
          )pb")));
}

TEST_F(GcsGrpcTest, RapidWriteWithOptions) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(
                  requests.back().write_offset() +
                  requests.back().checksummed_data().content().size());
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name(
                    requests.front().write_object_spec().resource().name());
                res->set_bucket(
                    requests.front().write_object_spec().resource().bucket());
                res->set_generation(100);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  MoveObjectRequest move_request;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce([&](auto*, const MoveObjectRequest* req, auto* resp) {
        move_request = *req;
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(100);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord("abcd"),
                                  {StorageGeneration::FromUint64(17)})
                       .result());

  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(100));
  ASSERT_EQ(requests.size(), 2);
  EXPECT_THAT(requests[0].write_object_spec().resource().name(),
              testing::StartsWith("abc.__tmp_"));
  EXPECT_TRUE(requests[0].write_object_spec().appendable());
  EXPECT_EQ(requests[0].write_object_spec().object_size(), 4);
  EXPECT_TRUE(requests[0].state_lookup());
  EXPECT_FALSE(requests[0].finish_write());
  EXPECT_TRUE(requests[1].finish_write());

  EXPECT_EQ(move_request.destination_object(), "abc");
  EXPECT_EQ(move_request.source_object(),
            requests[0].write_object_spec().resource().name());
  EXPECT_EQ(move_request.if_generation_match(), 17);
}

TEST_F(GcsGrpcTest, RapidWriteEmpty) {
  std::vector<BidiWriteObjectRequest> requests;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(
                  requests.back().write_offset() +
                  requests.back().checksummed_data().content().size());
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name(
                    requests.front().write_object_spec().resource().name());
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(1);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord()).result());

  EXPECT_THAT(
      requests,
      testing::AllOf(testing::SizeIs(testing::Ge(1)),
                     testing::Each(MatchesTempWriteRequest("abc.__tmp_", R"pb(
                       write_object_spec {
                         resource { bucket: "projects/_/buckets/bucket" }
                         object_size: 0
                         appendable: true
                       }
                       checksummed_data { crc32c: 0 }
                       object_checksums { crc32c: 0 }
                       finish_write: true
                       write_offset: 0
                     )pb"))));
}

TEST_F(GcsGrpcTest, RapidWriteObjectChecksums) {
  std::vector<BidiWriteObjectRequest> requests;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(
                  requests.back().write_offset() +
                  requests.back().checksummed_data().content().size());
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name("abc");
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(42);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord("abcd")).result());

  ASSERT_THAT(requests, testing::SizeIs(testing::Ge(1)));
  EXPECT_TRUE(requests.back().finish_write());
  EXPECT_TRUE(requests.back().has_object_checksums());
  EXPECT_EQ(requests.back().object_checksums().crc32c(), 2462583345);
}

// A bucket detected as regional uses the bidi streaming path without the
// appendable flag and without the temporary object.
TEST_F(GcsGrpcTest, BidiWriteKnownRegionalBucket) {
  std::vector<BidiWriteObjectRequest> requests;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            while (true) {
              BidiWriteObjectRequest req;
              if (!stream->Read(&req)) break;
              requests.push_back(std::move(req));
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(
                  requests.back().write_offset() +
                  requests.back().checksummed_data().content().size());
              if (requests.back().finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name("abc");
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(77);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenStore();
  TestSetIsZonal(store, false);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord("abcd")).result());

  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(77));
  ASSERT_THAT(requests, testing::SizeIs(testing::Ge(1)));
  EXPECT_TRUE(requests.back().finish_write());
  EXPECT_FALSE(requests.front().write_object_spec().appendable());
  EXPECT_TRUE(requests.back().has_object_checksums());
  EXPECT_EQ(requests.back().object_checksums().crc32c(), 2462583345);
}

// A 1024-byte key with multi-byte UTF-8 characters crossing the 985-byte
// truncation point produces a valid <= 1024-byte UTF-8 temporary object name
// with a 128-bit hex suffix and moves it to the full 1024-byte destination key.
TEST_F(GcsGrpcTest, AtomicZonalWriteTruncatesMaxKeyLengthOnUtf8Boundary) {
  // Place a 3-byte UTF-8 character ("\xE2\x82\xAC", U+20AC) starting at byte
  // 984 so byte 985 lands inside the code point, then pad to 1024 bytes.
  std::string long_key(984, 'a');
  long_key.append("\xE2\x82\xAC");
  long_key.append(1024 - long_key.size(), 'b');
  ASSERT_EQ(long_key.size(), 1024u);
  ASSERT_TRUE(tensorstore::internal_storage_gcs::IsValidObjectName(long_key));

  std::string captured_temp_name;
  std::string captured_dest_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            captured_temp_name = req.write_object_spec().resource().name();
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (req.finish_write()) {
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name(captured_temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(42);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce([&](auto*, const MoveObjectRequest* req, auto* resp) {
        captured_dest_name = req->destination_object();
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(99);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, long_key, absl::Cord("data")).result());

  // The 3-byte code point at 984..986 is dropped rather than split, leaving a
  // 984-byte prefix + 7-byte ".__tmp_" + 32-byte 128-bit hex = 1023 bytes.
  EXPECT_EQ(captured_temp_name.size(), 984u + 7u + 32u);
  EXPECT_LE(captured_temp_name.size(), 1024u);
  EXPECT_TRUE(
      tensorstore::internal_storage_gcs::IsValidObjectName(captured_temp_name));
  EXPECT_THAT(captured_temp_name, testing::StartsWith(absl::StrCat(
                                      std::string(984, 'a'), ".__tmp_")));
  EXPECT_EQ(captured_dest_name, long_key);
}

TEST_F(GcsGrpcTest, BidiWriteTestHook) {
  bool hook_called = false;
  tensorstore::internal_gcs_grpc::SetTestHook(
      [&](grpc::ClientContext* context, BidiWriteObjectRequest& req) {
        hook_called = true;
        EXPECT_TRUE(req.has_write_object_spec());
        context->AddMetadata("x-test-hook", "active");
      });
  struct ResetHookOnExit {
    ~ResetHookOnExit() {
      tensorstore::internal_gcs_grpc::SetTestHook(
          absl::AnyInvocable<void(grpc::ClientContext*,
                                  BidiWriteObjectRequest&)>());
    }
  } reset_hook;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto* context, auto* stream) -> grpc::Status {
        auto metadata = context->client_metadata();
        auto it = metadata.find("x-test-hook");
        EXPECT_NE(it, metadata.end());
        if (it != metadata.end()) {
          EXPECT_EQ(it->second, "active");
        }
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (req.finish_write()) {
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name("hooked");
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(77);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "hooked", absl::Cord("hello")).result());
  EXPECT_TRUE(hook_called);
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(77));
}

TEST_F(GcsGrpcTest, AtomicZonalWriteWorkflow) {
  std::string captured_temp_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            captured_temp_name = req.write_object_spec().resource().name();
            EXPECT_THAT(captured_temp_name,
                        testing::StartsWith("target.__tmp_"));
            EXPECT_TRUE(req.write_object_spec().appendable());
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (req.finish_write()) {
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name(captured_temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(42);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce([&](auto*, const MoveObjectRequest* req, auto* resp) {
        EXPECT_EQ(req->destination_object(), "target");
        EXPECT_EQ(req->source_object(), captured_temp_name);
        EXPECT_EQ(req->if_source_generation_match(), 42);
        resp->set_name("target");
        resp->set_bucket("projects/_/buckets/bucket");
        resp->set_generation(99);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
}

// `if_equal == NoValue()` means "the destination must not already exist",
// which `MoveObject` expresses as `if_generation_match: 0`.
TEST_F(GcsGrpcTest, AtomicZonalWriteIfNoValuePrecondition) {
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        std::string temp_name;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            temp_name = req.write_object_spec().resource().name();
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (req.finish_write()) {
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name(temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(42);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce([&](auto*, const MoveObjectRequest* req, auto* resp) {
        EXPECT_TRUE(req->has_if_generation_match());
        EXPECT_EQ(req->if_generation_match(), 0);
        resp->set_name("target");
        resp->set_bucket("projects/_/buckets/bucket");
        resp->set_generation(99);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "target", absl::Cord("data"),
                                  {StorageGeneration::NoValue()})
                       .result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
}

// A zonal write is one kvstore write: the write metrics are recorded exactly
// once, for the caller's payload, not once per underlying RPC.
TEST_F(GcsGrpcTest, AtomicZonalWriteRecordsMetricsOnce) {
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        std::string temp_name;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            temp_name = req.write_object_spec().resource().name();
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (req.finish_write()) {
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name(temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(42);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  auto get_bytes_written = []() -> int64_t {
    auto m = tensorstore::internal_metrics::GetMetricRegistry().Collect(
        "/tensorstore/kvstore/gcs_grpc/bytes_written");
    if (!m.has_value() || m->values.empty()) return 0;
    return std::get<int64_t>(m->values[0].value);
  };
  auto get_latency_count = []() -> int64_t {
    auto m = tensorstore::internal_metrics::GetMetricRegistry().Collect(
        "/tensorstore/kvstore/gcs_grpc/write_latency_ms");
    if (!m.has_value() || m->histograms.empty()) return 0;
    return static_cast<int64_t>(m->histograms[0].count);
  };
  const int64_t bytes_before = get_bytes_written();
  const int64_t latency_before = get_latency_count();

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(get_bytes_written(), bytes_before + 4);
  EXPECT_EQ(get_latency_count(), latency_before + 1);
}

TEST_F(GcsGrpcTest, Delete) {
  DeleteObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(grpc::Status::OK));

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
      .Times(AtLeast(1))
      .WillRepeatedly(Return(grpc::Status::OK));

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
      .Times(AtLeast(1))
      .WillRepeatedly(
          DoAll(SetArgPointee<2>(response1), Return(grpc::Status::OK)));

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(request2), _))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(grpc::Status::OK));

  EXPECT_CALL(mock(), DeleteObject(_, EqualsProto(request3), _))
      .Times(AtLeast(1))
      .WillRepeatedly(Return(grpc::Status::OK));

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
      .Times(AtLeast(1))
      .WillRepeatedly(
          DoAll(SetArgPointee<2>(response1), Return(grpc::Status::OK)));

  EXPECT_CALL(mock(), ListObjects(_, EqualsProto(request2), _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          DoAll(SetArgPointee<2>(response2), Return(grpc::Status::OK)));

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

TEST_F(GcsGrpcTest, UserProjectPlumbedWhenNotZonal) {
  EXPECT_CALL(mock(), ReadObject(_, _, _))
      .WillOnce([&](grpc::ServerContext* context, auto* req, auto* writer) {
        auto metadata = context->client_metadata();
        auto it = metadata.find("x-goog-user-project");
        EXPECT_NE(it, metadata.end());
        if (it != metadata.end()) {
          EXPECT_EQ(it->second, "my-billing-project");
        }
        writer->Write(ParseTextProtoOrDie(R"pb(
          metadata { generation: 2 }
          checksummed_data { content: '1234' }
        )pb"));
        return grpc::Status::OK;
      });

  auto store = kvstore::Open({{"driver", "gcs_grpc"},
                              {"endpoint", mock_service_.server_address()},
                              {"bucket", "bucket"},
                              {"timeout", "250ms"},
                              {"gcs_user_project",
                               {{"project_id", "my-billing-project"}}}})
                   .value();
  auto read_result = kvstore::Read(store, "a").result();
  TENSORSTORE_EXPECT_OK(read_result);
}

// Zonal buckets do not support requester pays, so the explicit combination is
// rejected when the store is opened rather than on every request.
TEST_F(GcsGrpcTest, UserProjectWithZonalModeFailsToOpen) {
  EXPECT_THAT(kvstore::Open({{"driver", "gcs_grpc"},
                             {"endpoint", mock_service_.server_address()},
                             {"bucket", "bucket"},
                             {"timeout", "250ms"},
                             {"experimental_mode", "zonal"},
                             {"gcs_user_project",
                              {{"project_id", "my-billing-project"}}}})
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("gcs_user_project")));
}

TEST_F(GcsGrpcTest, UserProjectWithNonAtomicModeFailsToOpen) {
  EXPECT_THAT(kvstore::Open({{"driver", "gcs_grpc"},
                             {"endpoint", mock_service_.server_address()},
                             {"bucket", "bucket"},
                             {"timeout", "250ms"},
                             {"experimental_mode", "non_atomic"},
                             {"gcs_user_project",
                              {{"project_id", "my-billing-project"}}}})
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("gcs_user_project")));
}

// A bucket detected as zonal still sends the user project; whether that is
// acceptable is left to the service.
TEST_F(GcsGrpcTest, UserProjectSentWhenDetectedZonal) {
  EXPECT_CALL(mock(), ReadObject(_, _, _))
      .WillOnce([&](grpc::ServerContext* context, auto* req, auto* writer) {
        auto metadata = context->client_metadata();
        auto it = metadata.find("x-goog-user-project");
        EXPECT_NE(it, metadata.end());
        if (it != metadata.end()) {
          EXPECT_EQ(it->second, "my-billing-project");
        }
        writer->Write(ParseTextProtoOrDie(R"pb(
          metadata { generation: 2 }
          checksummed_data { content: '1234' }
        )pb"));
        return grpc::Status::OK;
      });

  auto store = kvstore::Open({{"driver", "gcs_grpc"},
                              {"endpoint", mock_service_.server_address()},
                              {"bucket", "bucket"},
                              {"timeout", "250ms"},
                              {"gcs_user_project",
                               {{"project_id", "my-billing-project"}}}})
                   .value();
  TestSetIsZonal(store, true);
  auto read_result = kvstore::Read(store, "a").result();
  TENSORSTORE_EXPECT_OK(read_result);
}

TEST(GcsGrpcSpecTest, InvalidSpec) {
  auto context = Context::Default();

  // Test with missing `"bucket"` key.
  EXPECT_THAT(kvstore::Open({{"driver", "gcs_grpc"}}, context).result(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"bucket"` key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "gcs_grpc"}, {"bucket", "bucket:xyz"}}, context)
          .result(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Test with invalid `"path"`
  EXPECT_THAT(
      kvstore::Open(
          {{"driver", "gcs_grpc"}, {"bucket", "my-bucket"}, {"path", "a\tb"}},
          context)
          .result(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid GCS path")));
}

TEST(GcsGrpcSpecTest, ModeSpecRoundTrip) {
  {
    ::nlohmann::json json_spec = {{"driver", "gcs_grpc"},
                                  {"bucket", "bucket"},
                                  {"experimental_mode", "standard"}};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                     kvstore::Spec::FromJson(json_spec));
    EXPECT_THAT(spec.ToJson(), IsOkAndHolds(MatchesJson(json_spec)));
  }
  {
    ::nlohmann::json json_spec = {{"driver", "gcs_grpc"},
                                  {"bucket", "bucket"},
                                  {"experimental_mode", "zonal"}};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                     kvstore::Spec::FromJson(json_spec));
    EXPECT_THAT(spec.ToJson(), IsOkAndHolds(MatchesJson(json_spec)));
  }
  {
    ::nlohmann::json json_spec = {{"driver", "gcs_grpc"},
                                  {"bucket", "bucket"},
                                  {"experimental_mode", "non_atomic"}};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                     kvstore::Spec::FromJson(json_spec));
    EXPECT_THAT(spec.ToJson(), IsOkAndHolds(MatchesJson(json_spec)));
  }
}

TEST(GcsGrpcSpecTest, ModeSpecOmitted) {
  ::nlohmann::json json_spec = {{"driver", "gcs_grpc"}, {"bucket", "bucket"}};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   kvstore::Spec::FromJson(json_spec));
  EXPECT_THAT(spec.ToJson(), IsOkAndHolds(MatchesJson(json_spec)));
}

TEST(GcsGrpcSpecTest, ModeSpecInvalid) {
  EXPECT_THAT(kvstore::Spec::FromJson({{"driver", "gcs_grpc"},
                                       {"bucket", "bucket"},
                                       {"experimental_mode", "invalid"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Spec::FromJson({{"driver", "gcs_grpc"},
                                       {"bucket", "bucket"},
                                       {"experimental_mode", true}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Zonal-bucket detection has no structured signal to rely on, so it matches
// service error text.  Pin every accepted variant and the status-code gate:
// if the service rewords a message, this test is where it should show up.
TEST(GcsGrpcErrorClassificationTest, IsZonalObjectRequiredError) {
  EXPECT_TRUE(IsZonalObjectRequiredError(
      absl::InvalidArgumentError("This bucket requires appendable objects.")));
  EXPECT_TRUE(IsZonalObjectRequiredError(absl::InvalidArgumentError(
      "The bucket type only supports appendable objects.")));

  // Only INVALID_ARGUMENT carries this signal.
  EXPECT_FALSE(IsZonalObjectRequiredError(
      absl::UnimplementedError("This bucket requires appendable objects.")));
  EXPECT_FALSE(IsZonalObjectRequiredError(
      absl::PermissionDeniedError("This bucket requires appendable objects.")));

  // Near misses.
  EXPECT_FALSE(IsZonalObjectRequiredError(absl::OkStatus()));
  EXPECT_FALSE(
      IsZonalObjectRequiredError(absl::InvalidArgumentError("bad request")));
  EXPECT_FALSE(IsZonalObjectRequiredError(
      absl::InvalidArgumentError("appendable objects are required")));
}

TEST(GcsGrpcErrorClassificationTest, IsAppendableNotSupportedError) {
  EXPECT_TRUE(IsAppendableNotSupportedError(absl::UnimplementedError(
      "Appendable objects are not supported for this request.")));
  EXPECT_TRUE(IsAppendableNotSupportedError(absl::InvalidArgumentError(
      "Appendable objects are not supported for this request.")));

  EXPECT_FALSE(IsAppendableNotSupportedError(absl::PermissionDeniedError(
      "Appendable objects are not supported for this request.")));
  EXPECT_FALSE(IsAppendableNotSupportedError(absl::OkStatus()));
  EXPECT_FALSE(
      IsAppendableNotSupportedError(absl::UnimplementedError("not supported")));
}

TEST(GcsGrpcUrlTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "gcs_grpc"}, {"bucket", "my-bucket"}, {"path", "abc"}},
      "gcs_grpc://my-bucket/abc");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "gcs_grpc"}, {"bucket", "my-bucket"}, {"path", "abc def"}},
      "gcs_grpc://my-bucket/abc%20def");
}

TEST(GcsGrpcUrlTest, ModeBucketToUrl) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson({{"driver", "gcs_grpc"},
                                          {"bucket", "my-bucket"},
                                          {"path", "abc"},
                                          {"experimental_mode", "zonal"}}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto url, spec.ToUrl());
  EXPECT_EQ(url, "gcs_grpc://my-bucket/abc");
}

TEST(GcsGrpcUrlTest, InvalidUri) {
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc:foo"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc:///"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket:xyz"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid GCS bucket name: \"bucket:xyz\"")));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket?query"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Query string not supported")));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket#fragment"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Fragment identifier not supported")));
  EXPECT_THAT(kvstore::Spec::FromUrl("gcs_grpc://bucket/a%0Ab"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid GCS path")));
}

TEST(GcsGrpcSecurityTest, CrossTenantStubPoolIsolation) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store1,
      kvstore::Open(
          {{"driver", "gcs_grpc"},
           {"endpoint", "localhost:12345"},
           {"bucket", "bucket"},
           {"num_channels", 1},
           {"context",
            {{"experimental_gcs_grpc_credentials", {{"type", "insecure"}}}}}})
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2,
      kvstore::Open(
          {{"driver", "gcs_grpc"},
           {"endpoint", "localhost:12345"},
           {"bucket", "bucket"},
           {"num_channels", 1},
           {"context",
            {{"experimental_gcs_grpc_credentials",
              {{"type", "access_token"}, {"access_token", "secret-token"}}}}}})
          .result());

  auto* driver1 =
      static_cast<tensorstore::internal_gcs_grpc::GcsGrpcKeyValueStore*>(
          store1.driver.get());
  auto* driver2 =
      static_cast<tensorstore::internal_gcs_grpc::GcsGrpcKeyValueStore*>(
          store2.driver.get());
  EXPECT_NE(driver1->auth_strategy_, driver2->auth_strategy_);
  EXPECT_NE(driver1->storage_stub_pool_, driver2->storage_stub_pool_);
}

TEST(GcsGrpcSecurityTest, DefaultAuthenticationStrategyPortAndAuthority) {
  using ::tensorstore::internal_gcs_grpc::
      CreateDefaultGrpcAuthenticationStrategy;
  auto google_strategy =
      CreateDefaultGrpcAuthenticationStrategy("storage.googleapis.com");
  auto localhost_strategy =
      CreateDefaultGrpcAuthenticationStrategy("localhost:12345");

  EXPECT_NE(google_strategy, localhost_strategy);
  EXPECT_EQ(
      CreateDefaultGrpcAuthenticationStrategy("storage.googleapis.com:443"),
      google_strategy);
  EXPECT_EQ(CreateDefaultGrpcAuthenticationStrategy(
                "dns:///storage.googleapis.com:443"),
            google_strategy);
  EXPECT_EQ(CreateDefaultGrpcAuthenticationStrategy(
                "dns://attacker-dns:53/storage.googleapis.com"),
            localhost_strategy);
}

TEST_F(GcsGrpcTest, BidiWriteReusesCordExpectedChecksum) {
  absl::Cord payload("precomputed-crc-payload");
  constexpr uint32_t kPrecomputedCrc = 0x12345678u;
  payload.SetExpectedChecksum(kPrecomputedCrc);

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        EXPECT_EQ(req.checksummed_data().crc32c(), kPrecomputedCrc);
        EXPECT_EQ(req.object_checksums().crc32c(), kPrecomputedCrc);
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("key");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(1);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore(false);
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "key", payload).result());
}

TEST_F(GcsGrpcTest, ReadFullObjectChecksumValidatedWithoutFragmentCrc32c) {
  EXPECT_CALL(mock(), ReadObject(_, _, _))
      .WillOnce(
          [](auto*, auto*,
             grpc::ServerWriter<ReadObjectResponse>* resp) -> grpc::Status {
            resp->Write(ParseTextProtoOrDie(R"pb(
              metadata { generation: 2 size: 4 }
              object_checksums { crc32c: 99999999 }
              checksummed_data { content: "1234" }
            )pb"));
            return grpc::Status::OK;
          });

  auto store = OpenStore();
  EXPECT_THAT(kvstore::Read(store, "abc").result(),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("does not match expected crc32c")));
}

TEST_F(GcsGrpcTest, SupportedFeaturesByMode) {
  auto standard_store = OpenModeStore("standard");
  EXPECT_EQ(standard_store.driver->GetSupportedFeatures(KeyRange()),
            SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
                SupportedFeatures::kAtomicWriteWithoutOverwrite);

  auto zonal_store = OpenModeStore("zonal");
  EXPECT_EQ(zonal_store.driver->GetSupportedFeatures(KeyRange()),
            SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
                SupportedFeatures::kAtomicWriteWithoutOverwrite);

  auto non_atomic_store = OpenModeStore("non_atomic");
  EXPECT_EQ(non_atomic_store.driver->GetSupportedFeatures(KeyRange()),
            SupportedFeatures::kNone);
}

}  // namespace
