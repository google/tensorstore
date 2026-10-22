// Copyright 2026 The TensorStore Authors
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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/internal/grpc/grpc_mock.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/gcs_grpc/mock_storage_service.h"
#include "tensorstore/kvstore/gcs_grpc/test_hook.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

// protos
#include "google/rpc/status.pb.h"
#include "google/storage/v2/storage.pb.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::Future;
using ::tensorstore::KvStore;
using ::tensorstore::ParseTextProtoOrDie;
using ::tensorstore::StatusIs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::grpc_mocker::MockGrpcServer;
using ::tensorstore::internal::AbslStatusToGrpcStatus;
using ::tensorstore::internal_gcs_grpc::TestGetIsZonal;
using ::tensorstore::internal_gcs_grpc::TestSetIsZonal;
using ::tensorstore_grpc::MatchesTempWriteRequest;
using ::tensorstore_grpc::MockStorage;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::HasSubstr;
using ::testing::Return;

using ::google::storage::v2::BidiWriteObjectRequest;
using ::google::storage::v2::BidiWriteObjectResponse;
using ::google::storage::v2::DeleteObjectRequest;
using ::google::storage::v2::GetObjectRequest;
using ::google::storage::v2::MoveObjectRequest;
using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::ReadObjectResponse;

std::string GetClientMetadata(grpc::ServerContext* context,
                              std::string_view key) {
  const auto& metadata = context->client_metadata();
  auto it = metadata.find(grpc::string_ref(key.data(), key.size()));
  return it == metadata.end()
             ? std::string()
             : std::string(it->second.data(), it->second.size());
}

std::string GetIdempotencyToken(grpc::ServerContext* context) {
  return GetClientMetadata(context, "x-goog-gcs-idempotency-token");
}

class GcsMockTest : public testing::Test {
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

  // Each test in this file should install and use its own expectations
  // on this mock service.
  MockStorage& mock() { return *mock_service_.service(); }

  MockGrpcServer<MockStorage> mock_service_;
};

TEST_F(GcsMockTest, ReadRetriesAfterResourceExhausted) {
  // Tests that a ReadObject request retries after transient RESOURCE_EXHAUSTED
  // errors and returns the object payload once the stream succeeds.
  ReadObjectRequest expected_request = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'abc'
  )pb");

  ::testing::Sequence s1;
  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(2)
      .InSequence(s1)
      .WillRepeatedly(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))));

  EXPECT_CALL(mock(), ReadObject(_, EqualsProto(expected_request), _))
      .Times(AtLeast(1))
      .InSequence(s1)
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

  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value, "1234");
  EXPECT_GT(result.stamp.time, start);
  EXPECT_EQ(result.stamp.generation, StorageGeneration::FromUint64(2));
}

TEST_F(GcsMockTest, WriteRetriesAfterResourceExhausted) {
  // Tests that a non-zonal BidiWriteObject upload retries from the initial
  // state_lookup probe after transient RESOURCE_EXHAUSTED errors.
  std::vector<BidiWriteObjectRequest> requests;

  ::testing::Sequence s1;
  EXPECT_CALL(mock(), BidiWriteObject)
      .InSequence(s1)
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))));

  EXPECT_CALL(mock(), BidiWriteObject)
      .InSequence(s1)
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (stream->Read(&req)) {
          requests.push_back(req);
        }
        return AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""));
      });

  EXPECT_CALL(mock(), BidiWriteObject)
      .Times(AtLeast(1))
      .InSequence(s1)
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
        res->set_bucket("bucket");
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

TEST_F(GcsMockTest, AtomicZonalWriteRetriesTempUploadAfterResourceExhausted) {
  // Tests that an atomic zonal write retries its temporary object upload after
  // transient RESOURCE_EXHAUSTED errors while reusing the same temporary name.
  std::vector<BidiWriteObjectRequest> requests;

  ::testing::Sequence s1;
  EXPECT_CALL(mock(), BidiWriteObject)
      .InSequence(s1)
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""))));

  EXPECT_CALL(mock(), BidiWriteObject)
      .InSequence(s1)
      .WillOnce(
          [&](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                              BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            BidiWriteObjectRequest req;
            if (stream->Read(&req)) {
              requests.push_back(req);
            }
            return AbslStatusToGrpcStatus(absl::ResourceExhaustedError(""));
          });

  EXPECT_CALL(mock(), BidiWriteObject)
      .Times(AtLeast(1))
      .InSequence(s1)
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

TEST_F(GcsMockTest, BidiWriteMissingResourceReturnsInternalError) {
  // Tests that a BidiWriteObject stream completing with OK but without a
  // resource message is rejected with an INTERNAL error.
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                             BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(req.write_offset() +
                                      req.checksummed_data().content().size());
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  EXPECT_THAT(
      kvstore::Write(store, "abc", absl::Cord("abcd")).result(),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("without resource")));
}

TEST_F(GcsMockTest, BidiWriteZeroGenerationReturnsInternalError) {
  // Tests that a BidiWriteObject stream completing with a zero generation in
  // its resource metadata is rejected with an INTERNAL error.
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](auto*, grpc::ServerReaderWriter<BidiWriteObjectResponse,
                                             BidiWriteObjectRequest>* stream)
              -> grpc::Status {
            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              BidiWriteObjectResponse resp;
              resp.set_persisted_size(req.write_offset() +
                                      req.checksummed_data().content().size());
              if (req.finish_write()) {
                auto* res = resp.mutable_resource();
                res->set_name("abc");
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(0);
              }
              stream->Write(resp);
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  EXPECT_THAT(
      kvstore::Write(store, "abc", absl::Cord("abcd")).result(),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("zero generation")));
}

TEST_F(GcsMockTest, UnconditionalTempUploadFailedPreconditionPropagatesError) {
  // Tests that a FAILED_PRECONDITION error during the temporary object upload
  // propagates as an error rather than being treated as a CAS race.
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::FailedPreconditionError("Storage class mismatch"))));

  auto store = OpenRapidStore();
  EXPECT_THAT(kvstore::Write(store, "abc", absl::Cord("abcd")).result(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Storage class mismatch")));
}

TEST_F(GcsMockTest, BidiWriteFollowsRedirectWithRoutingToken) {
  // Tests that when BidiWriteObject returns ABORTED with a
  // BidiWriteObjectRedirectedError, the driver restarts the upload with the
  // provided routing_token in x-goog-request-params targeting the same
  // temporary object.
  std::vector<std::string> request_params;
  std::vector<std::string> temp_names;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly(
          [&](grpc::ServerContext* context, auto* stream) -> grpc::Status {
            request_params.push_back(
                GetClientMetadata(context, "x-goog-request-params"));
            const bool redirect = request_params.size() == 1;

            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              if (req.has_write_object_spec()) {
                temp_names.push_back(req.write_object_spec().resource().name());
              }
              if (redirect) {
                EXPECT_TRUE(req.state_lookup());
                EXPECT_TRUE(req.checksummed_data().content().empty());
                EXPECT_FALSE(req.finish_write());
                break;
              }
              if (req.finish_write()) {
                BidiWriteObjectResponse resp;
                auto* res = resp.mutable_resource();
                res->set_name(temp_names.back());
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(42);
                stream->Write(resp);
              }
            }
            if (redirect) {
              google::storage::v2::BidiWriteObjectRedirectedError details;
              details.set_routing_token("tok-1");
              google::rpc::Status rpc_status;
              rpc_status.set_code(static_cast<int>(grpc::StatusCode::ABORTED));
              rpc_status.set_message("redirect");
              rpc_status.add_details()->PackFrom(details);
              return grpc::Status(grpc::StatusCode::ABORTED, "redirect",
                                  rpc_status.SerializeAsString());
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "abc", absl::Cord("abcd")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(42));

  ASSERT_EQ(request_params.size(), 2);
  EXPECT_THAT(request_params[0], testing::Not(HasSubstr("routing_token")));
  EXPECT_THAT(request_params[1], HasSubstr("&routing_token=tok-1"));
  ASSERT_EQ(temp_names.size(), 2);
  EXPECT_EQ(temp_names[0], temp_names[1]);
}

TEST_F(GcsMockTest, BidiWriteCachesRoutingTokenAcrossConcurrentWrites) {
  // Tests that a routing token learned from BidiWriteObjectRedirectedError on
  // the initial probe write is cached on the driver and reused by concurrent
  // waiting writes so only the initial probe is redirected.
  absl::Mutex mu;
  std::vector<std::string> request_params;
  absl::Notification first_request_started;
  absl::Notification release_first_request;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(4)
      .WillRepeatedly(
          [&](grpc::ServerContext* context, auto* stream) -> grpc::Status {
            std::string params =
                GetClientMetadata(context, "x-goog-request-params");
            bool is_first = false;
            {
              absl::MutexLock lock(mu);
              is_first = request_params.empty();
              request_params.push_back(params);
            }
            if (is_first) {
              first_request_started.Notify();
              release_first_request.WaitForNotification();
            }

            std::string temp_name;
            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              if (req.has_write_object_spec()) {
                temp_name = req.write_object_spec().resource().name();
              }
              if (is_first) {
                EXPECT_TRUE(req.state_lookup());
                EXPECT_TRUE(req.checksummed_data().content().empty());
                EXPECT_FALSE(req.finish_write());
                break;
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
            if (is_first) {
              google::storage::v2::BidiWriteObjectRedirectedError details;
              details.set_routing_token("tok-cached");
              google::rpc::Status rpc_status;
              rpc_status.set_code(static_cast<int>(grpc::StatusCode::ABORTED));
              rpc_status.set_message("redirect");
              rpc_status.add_details()->PackFrom(details);
              return grpc::Status(grpc::StatusCode::ABORTED, "redirect",
                                  rpc_status.SerializeAsString());
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  auto f0 = kvstore::Write(store, "k0", absl::Cord("v0"));
  first_request_started.WaitForNotification();

  auto f1 = kvstore::Write(store, "k1", absl::Cord("v1"));
  auto f2 = kvstore::Write(store, "k2", absl::Cord("v2"));
  release_first_request.Notify();

  TENSORSTORE_ASSERT_OK(f0.result());
  TENSORSTORE_ASSERT_OK(f1.result());
  TENSORSTORE_ASSERT_OK(f2.result());

  absl::MutexLock lock(mu);
  EXPECT_THAT(request_params,
              testing::ElementsAre(testing::Not(HasSubstr("routing_token")),
                                   HasSubstr("&routing_token=tok-cached"),
                                   HasSubstr("&routing_token=tok-cached"),
                                   HasSubstr("&routing_token=tok-cached")));
}

TEST_F(GcsMockTest, AutoDetectZonalWriteHoldsProbeAcrossFallbackRedirect) {
  // Tests that when bucket mode is unknown and the initial non-appendable
  // probe is redirected before the backend rejects appendable=false, concurrent
  // writes remain gated until is_zonal is resolved and then immediately use the
  // cached routing token without a second state_lookup probe.
  absl::Mutex mu;
  std::vector<std::string> request_params;
  std::vector<bool> appendable_flags;
  absl::Notification first_request_started;
  absl::Notification release_first_request;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(5)
      .WillRepeatedly([&](grpc::ServerContext* context,
                          auto* stream) -> grpc::Status {
        std::string params =
            GetClientMetadata(context, "x-goog-request-params");
        size_t call_index = 0;
        {
          absl::MutexLock lock(mu);
          call_index = request_params.size();
          request_params.push_back(params);
        }
        if (call_index == 0) {
          first_request_started.Notify();
          release_first_request.WaitForNotification();
        }

        std::string obj_name;
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            obj_name = req.write_object_spec().resource().name();
            absl::MutexLock lock(mu);
            appendable_flags.push_back(req.write_object_spec().appendable());
          }
          if (call_index <= 1) {
            EXPECT_FALSE(req.write_object_spec().appendable());
            EXPECT_TRUE(req.state_lookup());
            EXPECT_TRUE(req.checksummed_data().content().empty());
            EXPECT_FALSE(req.finish_write());
            break;
          }
          EXPECT_TRUE(req.write_object_spec().appendable());
          EXPECT_FALSE(req.state_lookup());
          EXPECT_TRUE(req.finish_write());
          BidiWriteObjectResponse resp;
          auto* res = resp.mutable_resource();
          res->set_name(obj_name);
          res->set_bucket("projects/_/buckets/bucket");
          res->set_generation(42 + call_index);
          stream->Write(resp);
        }
        if (call_index == 0) {
          google::storage::v2::BidiWriteObjectRedirectedError details;
          details.set_routing_token("tok-auto");
          google::rpc::Status rpc_status;
          rpc_status.set_code(static_cast<int>(grpc::StatusCode::ABORTED));
          rpc_status.set_message("redirect");
          rpc_status.add_details()->PackFrom(details);
          return grpc::Status(grpc::StatusCode::ABORTED, "redirect",
                              rpc_status.SerializeAsString());
        }
        if (call_index == 1) {
          return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              "The bucket requires appendable objects.");
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .Times(3)
      .WillRepeatedly([](auto*, const MoveObjectRequest* req, auto* resp) {
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(100);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  auto f0 = kvstore::Write(store, "k0", absl::Cord("v0"));
  first_request_started.WaitForNotification();

  auto f1 = kvstore::Write(store, "k1", absl::Cord("v1"));
  auto f2 = kvstore::Write(store, "k2", absl::Cord("v2"));
  release_first_request.Notify();

  TENSORSTORE_ASSERT_OK(f0.result());
  TENSORSTORE_ASSERT_OK(f1.result());
  TENSORSTORE_ASSERT_OK(f2.result());

  absl::MutexLock lock(mu);
  EXPECT_THAT(request_params,
              testing::ElementsAre(testing::Not(HasSubstr("routing_token")),
                                   HasSubstr("&routing_token=tok-auto"),
                                   HasSubstr("&routing_token=tok-auto"),
                                   HasSubstr("&routing_token=tok-auto"),
                                   HasSubstr("&routing_token=tok-auto")));
  EXPECT_THAT(appendable_flags,
              testing::ElementsAre(false, false, true, true, true));
}

TEST_F(GcsMockTest, BidiWriteInvalidatesCachedRoutingTokenOnRetriableError) {
  // Tests that when a write using a cached routing token fails with a
  // retriable error (UNAVAILABLE), the driver invalidates the cached token so
  // the retry reconnects without the stale token.
  std::vector<std::string> request_params;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(4)
      .WillRepeatedly(
          [&](grpc::ServerContext* context, auto* stream) -> grpc::Status {
            request_params.push_back(
                GetClientMetadata(context, "x-goog-request-params"));
            const size_t call_index = request_params.size() - 1;

            std::string temp_name;
            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              if (req.has_write_object_spec()) {
                temp_name = req.write_object_spec().resource().name();
              }
              if (call_index == 0 || call_index == 2) {
                break;
              }
              if (req.finish_write()) {
                BidiWriteObjectResponse resp;
                auto* res = resp.mutable_resource();
                res->set_name(temp_name);
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(42 + call_index);
                stream->Write(resp);
              }
            }
            if (call_index == 0) {
              google::storage::v2::BidiWriteObjectRedirectedError details;
              details.set_routing_token("tok-1");
              google::rpc::Status rpc_status;
              rpc_status.set_code(static_cast<int>(grpc::StatusCode::ABORTED));
              rpc_status.set_message("redirect");
              rpc_status.add_details()->PackFrom(details);
              return grpc::Status(grpc::StatusCode::ABORTED, "redirect",
                                  rpc_status.SerializeAsString());
            }
            if (call_index == 2) {
              return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                  "target unavailable");
            }
            return grpc::Status::OK;
          });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "k0", absl::Cord("v0")).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "k1", absl::Cord("v1")).result());

  ASSERT_EQ(request_params.size(), 4);
  EXPECT_THAT(request_params[0], testing::Not(HasSubstr("routing_token")));
  EXPECT_THAT(request_params[1], HasSubstr("&routing_token=tok-1"));
  EXPECT_THAT(request_params[2], HasSubstr("&routing_token=tok-1"));
  EXPECT_THAT(request_params[3], testing::Not(HasSubstr("routing_token")));
}

TEST_F(GcsMockTest, ZonalWriteFailsAndPinsModeWhenAppendableUnsupported) {
  // Tests that when a bucket detected as zonal rejects appendable objects, the
  // driver abandons the temporary object, pins is_zonal to false, and fails
  // immediately.
  std::string captured_temp_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (!stream->Read(&req)) {
          return grpc::Status(grpc::StatusCode::INTERNAL, "missing req");
        }
        EXPECT_TRUE(req.write_object_spec().appendable());
        captured_temp_name = req.write_object_spec().resource().name();
        EXPECT_THAT(captured_temp_name, testing::StartsWith("abc.__tmp_"));
        return grpc::Status(
            grpc::StatusCode::UNIMPLEMENTED,
            "Appendable objects are not supported for this request.");
      });

  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(AtLeast(0))
      .WillRepeatedly(Return(grpc::Status::OK));

  auto store = OpenStore();
  TestSetIsZonal(store, true);
  EXPECT_THAT(kvstore::Write(store, "abc", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Appendable objects are not supported")));
  EXPECT_EQ(TestGetIsZonal(store), false);
}

TEST_F(GcsMockTest, ZonalWriteFailsAndRetainsZonalModeWhenMoveUnsupported) {
  // Tests that when a bucket detected as zonal accepts appendable objects but
  // rejects MoveObject as unsupported, the driver deletes the temporary
  // object, retains is_zonal as true (since the appendable write succeeded),
  // and reports the failure.
  std::string captured_temp_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (captured_temp_name.empty()) {
            captured_temp_name = req.write_object_spec().resource().name();
            EXPECT_THAT(captured_temp_name, testing::StartsWith("abc.__tmp_"));
          }
          BidiWriteObjectResponse resp;
          resp.set_persisted_size(req.write_offset() +
                                  req.checksummed_data().content().size());
          if (req.finish_write()) {
            auto* res = resp.mutable_resource();
            res->set_name(captured_temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(7);
          }
          stream->Write(resp);
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce(Return(
          grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                       "The bucket does not support hierarchical namespace.")));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [&](auto*, const DeleteObjectRequest* req, auto*) -> grpc::Status {
            EXPECT_EQ(req->object(), captured_temp_name);
            if (!deleted.HasBeenNotified()) deleted.Notify();
            return grpc::Status::OK;
          });

  auto store = OpenStore();
  TestSetIsZonal(store, true);
  EXPECT_THAT(kvstore::Write(store, "abc", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("does not support hierarchical namespace")));

  ASSERT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_EQ(TestGetIsZonal(store), true);
}

TEST_F(GcsMockTest, ZonalWriteMoveUnsupportedDoesNotOverrideExplicitMode) {
  // Tests that when "mode": "zonal" is explicitly configured and MoveObject
  // fails as unsupported, the temporary object is cleaned up and the error is
  // returned without overriding the configured mode.
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
            res->set_generation(7);
            stream->Write(resp);
          }
        }
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce(Return(
          grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                       "The bucket does not support hierarchical namespace.")));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest*, auto*) {
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  EXPECT_THAT(kvstore::Write(store, "abc", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ASSERT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_EQ(TestGetIsZonal(store), true);
}

TEST_F(GcsMockTest, LearnedRegionalModeIsReusedAcrossSubsequentWrites) {
  // Tests that once a non-appendable write succeeds, the bucket is recorded as
  // regional (is_zonal == false) and subsequent writes use the direct
  // non-appendable path without MoveObject.
  EXPECT_CALL(mock(), MoveObject(_, _, _)).Times(0);

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(AtLeast(2))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            EXPECT_FALSE(req.write_object_spec().appendable());
            EXPECT_EQ(req.write_object_spec().resource().name(), "abc");
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("abc");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(99);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result1, kvstore::Write(store, "abc", absl::Cord("data1")).result());
  EXPECT_EQ(result1.generation, StorageGeneration::FromUint64(99));
  EXPECT_EQ(TestGetIsZonal(store), false);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result2, kvstore::Write(store, "abc", absl::Cord("data2")).result());
  EXPECT_EQ(result2.generation, StorageGeneration::FromUint64(99));
}

TEST_F(GcsMockTest, AutoDetectZonalSerializesSingleProbeAndSwitchesAllWrites) {
  // Tests that when bucket mode is unknown, concurrent writes serialize behind
  // a single non-appendable probe write, and once the probe detects that
  // appendable objects are required, both the probe and all waiting writes
  // complete via the atomic zonal path.
  std::atomic<int> non_appendable_count = 0;
  std::atomic<int> appendable_count = 0;
  std::atomic<int> move_count = 0;

  absl::Notification probe_started;
  absl::Notification release_probe;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (!stream->Read(&req)) {
          return grpc::Status(grpc::StatusCode::INTERNAL, "missing req");
        }
        if (!req.write_object_spec().appendable()) {
          EXPECT_TRUE(req.state_lookup());
          EXPECT_TRUE(req.checksummed_data().content().empty());
          EXPECT_FALSE(req.finish_write());
          non_appendable_count.fetch_add(1);
          if (!probe_started.HasBeenNotified()) {
            probe_started.Notify();
          }
          release_probe.WaitForNotification();
          return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              "bucket requires appendable objects");
        }
        appendable_count.fetch_add(1);
        std::string temp_name = req.write_object_spec().resource().name();
        while (!req.finish_write()) {
          if (req.state_lookup()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (!stream->Read(&req)) break;
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name(temp_name);
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(42);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillRepeatedly([&](auto*, const MoveObjectRequest* req, auto* resp) {
        move_count.fetch_add(1);
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(42);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  auto f0 = kvstore::Write(store, "key0", absl::Cord("data0"));
  probe_started.WaitForNotification();

  auto f1 = kvstore::Write(store, "key1", absl::Cord("data1"));
  auto f2 = kvstore::Write(store, "key2", absl::Cord("data2"));
  auto f3 = kvstore::Write(store, "key3", absl::Cord("data3"));
  auto f4 = kvstore::Write(store, "key4", absl::Cord("data4"));

  release_probe.Notify();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r0, f0.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r1, f1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r2, f2.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r3, f3.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r4, f4.result());

  EXPECT_EQ(non_appendable_count.load(), 1);
  EXPECT_EQ(appendable_count.load(), 5);
  EXPECT_EQ(move_count.load(), 5);
  EXPECT_EQ(TestGetIsZonal(store), true);
}

TEST_F(GcsMockTest, AutoDetectRegionalSerializesSingleProbeAndCompletesDirect) {
  // Tests that when bucket mode is unknown and the initial non-appendable
  // probe succeeds, the probe and all waiting concurrent writes complete
  // directly via the non-appendable path without MoveObject.
  std::atomic<int> non_appendable_count = 0;
  EXPECT_CALL(mock(), MoveObject(_, _, _)).Times(0);

  absl::Notification probe_started;
  absl::Notification release_probe;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (!stream->Read(&req)) {
          return grpc::Status(grpc::StatusCode::INTERNAL, "missing req");
        }
        EXPECT_FALSE(req.write_object_spec().appendable());
        int count = non_appendable_count.fetch_add(1);
        if (count == 0) {
          EXPECT_TRUE(req.state_lookup());
          EXPECT_TRUE(req.checksummed_data().content().empty());
          EXPECT_FALSE(req.finish_write());
          if (!probe_started.HasBeenNotified()) {
            probe_started.Notify();
          }
          release_probe.WaitForNotification();
        }
        std::string name = req.write_object_spec().resource().name();
        while (!req.finish_write()) {
          if (req.state_lookup()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          if (!stream->Read(&req)) break;
        }
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name(name);
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(77);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  auto f0 = kvstore::Write(store, "key0", absl::Cord("data0"));
  probe_started.WaitForNotification();

  auto f1 = kvstore::Write(store, "key1", absl::Cord("data1"));
  auto f2 = kvstore::Write(store, "key2", absl::Cord("data2"));
  auto f3 = kvstore::Write(store, "key3", absl::Cord("data3"));
  auto f4 = kvstore::Write(store, "key4", absl::Cord("data4"));

  release_probe.Notify();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r0, f0.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r1, f1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r2, f2.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r3, f3.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r4, f4.result());

  EXPECT_EQ(non_appendable_count.load(), 5);
  EXPECT_EQ(TestGetIsZonal(store), false);
}

TEST_F(GcsMockTest, AutoDetectProbeCancellationWakesWaitingWrites) {
  // Tests that if the initial probe write is cancelled while concurrent writes
  // are waiting, releasing the probe wakes the waiting writes so one of them
  // takes over as the new probe and completes bucket-type detection.
  std::atomic<int> non_appendable_count = 0;
  std::atomic<int> appendable_count = 0;
  std::atomic<int> move_count = 0;

  absl::Notification first_probe_started;
  absl::Notification first_probe_cancelled;

  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillRepeatedly(
          [&](grpc::ServerContext* context, auto* stream) -> grpc::Status {
            BidiWriteObjectRequest req;
            if (!stream->Read(&req)) {
              return grpc::Status(grpc::StatusCode::INTERNAL, "missing req");
            }
            if (!req.write_object_spec().appendable()) {
              int count = non_appendable_count.fetch_add(1);
              if (count == 0) {
                first_probe_started.Notify();
                first_probe_cancelled.WaitForNotification();
                return grpc::Status(grpc::StatusCode::CANCELLED, "cancelled");
              }
              return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                  "bucket requires appendable objects");
            }
            appendable_count.fetch_add(1);
            std::string temp_name = req.write_object_spec().resource().name();
            while (!req.finish_write()) {
              if (req.state_lookup()) {
                BidiWriteObjectResponse state_resp;
                stream->Write(state_resp);
              }
              if (!stream->Read(&req)) break;
            }
            BidiWriteObjectResponse resp;
            auto* res = resp.mutable_resource();
            res->set_name(temp_name);
            res->set_bucket("projects/_/buckets/bucket");
            res->set_generation(42);
            stream->Write(resp);
            return grpc::Status::OK;
          });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillRepeatedly([&](auto*, const MoveObjectRequest* req, auto* resp) {
        move_count.fetch_add(1);
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(42);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  Future<tensorstore::TimestampedStorageGeneration> f1;
  Future<tensorstore::TimestampedStorageGeneration> f2;
  {
    auto f0 = kvstore::Write(store, "key0", absl::Cord("data0"));
    first_probe_started.WaitForNotification();

    f1 = kvstore::Write(store, "key1", absl::Cord("data1"));
    f2 = kvstore::Write(store, "key2", absl::Cord("data2"));
  }
  first_probe_cancelled.Notify();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r1, f1.result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r2, f2.result());
  EXPECT_EQ(r1.generation, StorageGeneration::FromUint64(42));
  EXPECT_EQ(r2.generation, StorageGeneration::FromUint64(42));
  EXPECT_EQ(non_appendable_count.load(), 2);
  EXPECT_EQ(appendable_count.load(), 2);
  EXPECT_EQ(move_count.load(), 2);
  EXPECT_EQ(TestGetIsZonal(store), true);
}

TEST_F(GcsMockTest, AtomicZonalWritePreconditionMismatchCleansTemp) {
  // Tests that when MoveObject fails with FAILED_PRECONDITION due to a
  // destination generation mismatch, the driver deletes the temporary object
  // and returns StorageGeneration::Unknown().
  std::string captured_temp_name;
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
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::FailedPreconditionError("Precondition failed"))));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "target", absl::Cord("data"),
                                  {StorageGeneration::FromUint64(10)})
                       .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(result.generation));
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(60)));
}

TEST_F(GcsMockTest, AtomicZonalWriteConditionalNotFoundPreconditionCleansTemp) {
  // Tests that on attempt 0, a NOT_FOUND from MoveObject with a specific
  // generation precondition indicates the destination object does not exist,
  // deletes the temporary object, and returns StorageGeneration::Unknown().
  std::string captured_temp_name;
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
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::NotFoundError("Object not found"))));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "target", absl::Cord("data"),
                                  {StorageGeneration::FromUint64(10)})
                       .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(result.generation));
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(60)));
}

TEST_F(GcsMockTest, AtomicZonalWriteRetriedMoveReusesIdempotencyToken) {
  // Tests that when MoveObject is retried after a transient failure, both
  // attempts send the same non-empty x-goog-gcs-idempotency-token header.
  std::string captured_temp_name;
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

  std::string first_token;
  std::string second_token;
  ::testing::Sequence s;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          [&](grpc::ServerContext* context, const MoveObjectRequest*, auto*) {
            first_token = GetIdempotencyToken(context);
            return AbslStatusToGrpcStatus(
                absl::UnavailableError("transient failure"));
          });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](grpc::ServerContext* context, const MoveObjectRequest* req,
                    auto* resp) {
        second_token = GetIdempotencyToken(context);
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(99);
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
  EXPECT_FALSE(first_token.empty());
  EXPECT_EQ(first_token, second_token);
}

TEST_F(GcsMockTest, AtomicZonalWriteRetriedMoveNotFoundQueriesDestination) {
  // Tests that when MoveObject is retried after a transient error and returns
  // NOT_FOUND, MoveTask increments the move_fallback metric, queries the
  // destination via GetObject, and succeeds when the destination metadata
  // carries this write's tensorstore-write-id.
  std::string captured_temp_name;
  std::string captured_write_id;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            const auto& resource = req.write_object_spec().resource();
            captured_temp_name = resource.name();
            auto it = resource.metadata().find("tensorstore-write-id");
            if (it != resource.metadata().end()) {
              captured_write_id = it->second;
            }
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

  std::string idempotency_token;
  ::testing::Sequence s;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          [&](grpc::ServerContext* context, const MoveObjectRequest*, auto*) {
            idempotency_token = GetIdempotencyToken(context);
            return AbslStatusToGrpcStatus(
                absl::UnavailableError("transient failure"));
          });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::NotFoundError("source object not found"))));

  GetObjectRequest expected_get = ParseTextProtoOrDie(R"pb(
    bucket: 'projects/_/buckets/bucket'
    object: 'target'
  )pb");
  EXPECT_CALL(mock(), GetObject(_, EqualsProto(expected_get), _))
      .InSequence(s)
      .WillOnce([&](grpc::ServerContext* context, const GetObjectRequest* req,
                    auto* resp) {
        EXPECT_EQ(GetIdempotencyToken(context), "");
        resp->set_name(req->object());
        resp->set_bucket(req->bucket());
        resp->set_generation(99);
        (*resp->mutable_metadata())["tensorstore-write-id"] = captured_write_id;
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), DeleteObject(_, _, _)).Times(0);

  auto get_fallback_count = []() -> int64_t {
    auto m = tensorstore::internal_metrics::GetMetricRegistry().Collect(
        "/tensorstore/kvstore/gcs_grpc/move_fallback");
    if (!m.has_value() || m->values.empty()) return 0;
    return std::get<int64_t>(m->values[0].value);
  };
  const int64_t before = get_fallback_count();

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
  EXPECT_EQ(get_fallback_count(), before + 1);
  EXPECT_FALSE(captured_write_id.empty());
  EXPECT_EQ(captured_write_id, idempotency_token);
}

TEST_F(GcsMockTest, BidiWriteRetriedReusesIdempotencyToken) {
  // Tests that when a BidiWriteObject stream is retried after a transient
  // failure, both stream attempts send the same x-goog-gcs-idempotency-token.
  std::vector<std::string> tokens;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly(
          [&](grpc::ServerContext* context, auto* stream) -> grpc::Status {
            tokens.push_back(GetIdempotencyToken(context));
            BidiWriteObjectRequest req;
            while (stream->Read(&req)) {
              if (tokens.size() == 1) {
                break;
              }
              if (req.finish_write()) {
                BidiWriteObjectResponse resp;
                auto* res = resp.mutable_resource();
                res->set_name("target");
                res->set_bucket("projects/_/buckets/bucket");
                res->set_generation(55);
                stream->Write(resp);
              }
            }
            if (tokens.size() == 1) {
              return AbslStatusToGrpcStatus(
                  absl::UnavailableError("transient stream failure"));
            }
            return grpc::Status::OK;
          });

  auto store = OpenModeStore("standard");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(55));
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_FALSE(tokens[0].empty());
  EXPECT_EQ(tokens[0], tokens[1]);
}

TEST_F(GcsMockTest, DeleteRetriedReusesIdempotencyToken) {
  // Tests that when a DeleteObject RPC is retried after a transient failure,
  // both attempts send the same non-empty x-goog-gcs-idempotency-token.
  std::string first_token;
  std::string second_token;
  ::testing::Sequence s;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          [&](grpc::ServerContext* context, const DeleteObjectRequest*, auto*) {
            first_token = GetIdempotencyToken(context);
            return AbslStatusToGrpcStatus(
                absl::UnavailableError("transient delete failure"));
          });

  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          [&](grpc::ServerContext* context, const DeleteObjectRequest*, auto*) {
            second_token = GetIdempotencyToken(context);
            return grpc::Status::OK;
          });

  auto store = OpenModeStore("standard");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Delete(store, "target", {StorageGeneration::FromUint64(12)})
          .result());
  EXPECT_EQ(result.generation, StorageGeneration::NoValue());
  EXPECT_FALSE(first_token.empty());
  EXPECT_EQ(first_token, second_token);
}

TEST_F(GcsMockTest, AtomicZonalWriteRetriedMoveNotFoundDestinationMissing) {
  // Tests that when a retried MoveObject returns NOT_FOUND and GetObject on
  // the destination also returns NOT_FOUND, the driver deletes the temporary
  // object and returns StorageGeneration::Unknown().
  std::string captured_temp_name;
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

  ::testing::Sequence s;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(testing::Return(
          AbslStatusToGrpcStatus(absl::UnavailableError("transient failure"))));

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::NotFoundError("source object not found"))));

  EXPECT_CALL(mock(), GetObject(_, _, _))
      .InSequence(s)
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::NotFoundError("destination object not found"))));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result, kvstore::Write(store, "target", absl::Cord("data"),
                                  {StorageGeneration::FromUint64(10)})
                       .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(result.generation));
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(60)));
}

TEST_F(GcsMockTest, AtomicZonalWriteRetriedMoveNotFoundForeignWriteId) {
  // Tests that when a retried MoveObject returns NOT_FOUND and GetObject shows
  // a destination object carrying a different writer's tensorstore-write-id,
  // the driver deletes the temporary object and returns
  // StorageGeneration::Unknown() for a conditional write and ABORTED for an
  // unconditional write.
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly([](auto*, auto* stream) -> grpc::Status {
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

  ::testing::Sequence s;
  for (int i = 0; i < 2; ++i) {
    EXPECT_CALL(mock(), MoveObject(_, _, _))
        .InSequence(s)
        .WillOnce(testing::Return(AbslStatusToGrpcStatus(
            absl::UnavailableError("transient failure"))));

    EXPECT_CALL(mock(), MoveObject(_, _, _))
        .InSequence(s)
        .WillOnce(testing::Return(AbslStatusToGrpcStatus(
            absl::NotFoundError("source object not found"))));

    EXPECT_CALL(mock(), GetObject(_, _, _))
        .InSequence(s)
        .WillOnce([](auto*, const GetObjectRequest* req, auto* resp) {
          resp->set_name(req->object());
          resp->set_bucket(req->bucket());
          resp->set_generation(99);
          (*resp->mutable_metadata())["tensorstore-write-id"] = "someone-else";
          return grpc::Status::OK;
        });
  }

  std::atomic<int> delete_count = 0;
  absl::Notification all_deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(2)
      .WillRepeatedly([&](auto*, const DeleteObjectRequest*, auto*) {
        if (delete_count.fetch_add(1) + 1 == 2) {
          all_deleted.Notify();
        }
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto cond_result, kvstore::Write(store, "target", absl::Cord("data"),
                                       {StorageGeneration::FromUint64(10)})
                            .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(cond_result.generation));

  EXPECT_THAT(kvstore::Write(store, "target", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kAborted));
  EXPECT_TRUE(all_deleted.WaitForNotificationWithTimeout(absl::Seconds(60)));
}

TEST_F(GcsMockTest, AtomicZonalWriteMovePermissionDeniedPropagatesError) {
  // Tests that a non-precondition MoveObject failure (PERMISSION_DENIED)
  // deletes the temporary object and surfaces the error to the caller.
  std::string captured_temp_name;
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
      .WillOnce(testing::Return(AbslStatusToGrpcStatus(
          absl::PermissionDeniedError("storage.objects.move denied"))));

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  EXPECT_THAT(kvstore::Write(store, "target", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(60)));
}

TEST_F(GcsMockTest, AtomicZonalWriteCancelledDuringUploadCleansTemp) {
  // Tests that cancelling a zonal write while the temporary upload is in
  // flight skips MoveObject and deletes the created temporary generation.
  absl::Notification upload_started;
  absl::Notification release_upload;
  std::string captured_temp_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        bool finished = false;
        while (!finished && stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            captured_temp_name = req.write_object_spec().resource().name();
          }
          if (req.state_lookup() && !req.finish_write()) {
            BidiWriteObjectResponse state_resp;
            stream->Write(state_resp);
          }
          finished = req.finish_write();
        }
        upload_started.Notify();
        EXPECT_TRUE(
            release_upload.WaitForNotificationWithTimeout(absl::Seconds(10)));
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name(captured_temp_name);
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(42);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _)).Times(0);

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        EXPECT_TRUE(req->has_if_generation_match());
        EXPECT_EQ(req->if_generation_match(), 42);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  {
    auto write_future = kvstore::Write(store, "target", absl::Cord("data"));
    ASSERT_TRUE(
        upload_started.WaitForNotificationWithTimeout(absl::Seconds(10)));
  }
  release_upload.Notify();
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_F(GcsMockTest, AutoTransitionToAtomicZonalWriteOnAppendableRequired) {
  // Tests that when a store opened without an explicit mode receives an
  // appendable-required error on its initial probe, it transitions to zonal
  // mode and completes the write via a temporary object and MoveObject.
  std::string captured_temp_name;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (stream->Read(&req)) {
          EXPECT_FALSE(req.write_object_spec().appendable());
          EXPECT_TRUE(req.state_lookup());
          EXPECT_TRUE(req.checksummed_data().content().empty());
          EXPECT_FALSE(req.finish_write());
        }
        return AbslStatusToGrpcStatus(absl::InvalidArgumentError(
            "This bucket requires appendable objects."));
      })
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
        resp->set_name("target");
        resp->set_bucket("projects/_/buckets/bucket");
        resp->set_generation(99);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto result,
      kvstore::Write(store, "target", absl::Cord("data")).result());
  EXPECT_EQ(result.generation, StorageGeneration::FromUint64(99));
  EXPECT_EQ(TestGetIsZonal(store), true);
}

TEST_F(GcsMockTest, BidiWriteFailsFastOnRapidByteQuotaExhausted) {
  // Tests that RESOURCE_EXHAUSTED or UNAVAILABLE errors reporting Rapid
  // storage class byte quota exhaustion fail immediately on the first attempt
  // without retrying.
  ::testing::Sequence s;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .InSequence(s)
      .WillOnce(
          testing::Return(grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                       "You have exhausted your project's "
                                       "Rapid storage class byte quota.")));
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .InSequence(s)
      .WillOnce(
          testing::Return(grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                       "You have exhausted your project's "
                                       "Rapid storage class byte quota.")));

  auto store = OpenRapidStore(false);
  EXPECT_THAT(kvstore::Write(store, "key0", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("Rapid storage class byte quota")));
  EXPECT_THAT(kvstore::Write(store, "key1", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kUnavailable,
                       HasSubstr("Rapid storage class byte quota")));
}

TEST_F(GcsMockTest, ExplicitZonalModesCompleteProbeWhenAppendableNotSupported) {
  // Tests that when "mode" is explicitly set to "zonal" and the service
  // rejects appendable objects, the initial routing probe finishes on the
  // first failure so subsequent writes do not repeat the 0-byte state_lookup
  // probe.
  std::vector<bool> state_lookups;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        if (stream->Read(&req)) {
          state_lookups.push_back(req.state_lookup());
        }
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            "Appendable objects are not supported for this request.");
      });
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(AtLeast(0))
      .WillRepeatedly(Return(grpc::Status::OK));

  auto zonal_store = OpenModeStore("zonal");
  EXPECT_THAT(kvstore::Write(zonal_store, "k0", absl::Cord("v0")).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Appendable objects are not supported")));
  EXPECT_THAT(kvstore::Write(zonal_store, "k1", absl::Cord("v1")).result(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Appendable objects are not supported")));
  EXPECT_EQ(TestGetIsZonal(zonal_store), true);

  EXPECT_THAT(state_lookups, testing::ElementsAre(true, false));
}

TEST_F(GcsMockTest, DirectWriteNoValuePreconditionNotFoundPropagatesError) {
  // Tests that when if_equal is NoValue() (if_generation_match: 0), a
  // NOT_FOUND error from BidiWriteObject is propagated as an error rather than
  // masked as StorageGeneration::Unknown().
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        stream->Read(&req);
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "bucket not found");
      });

  EXPECT_THAT(kvstore::Write(OpenModeStore("standard"), "k0", absl::Cord("v0"),
                             {StorageGeneration::NoValue()})
                  .result(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(GcsMockTest,
       AtomicZonalWriteUnconditionalAndNoValueMoveNotFoundPropagatesError) {
  // Tests that on attempt 0, a NOT_FOUND from MoveObject when if_equal is
  // Unknown() or NoValue() does not query the destination and propagates
  // NOT_FOUND as an error while cleaning up the temporary object.
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly([](auto*, auto* stream) -> grpc::Status {
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
      .Times(2)
      .WillRepeatedly(
          Return(grpc::Status(grpc::StatusCode::NOT_FOUND, "not found")));
  EXPECT_CALL(mock(), GetObject(_, _, _)).Times(0);

  std::atomic<int> delete_count = 0;
  absl::Notification all_deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(2)
      .WillRepeatedly([&](auto*, const DeleteObjectRequest*, auto*) {
        if (delete_count.fetch_add(1) + 1 == 2) {
          all_deleted.Notify();
        }
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  EXPECT_THAT(kvstore::Write(store, "k0", absl::Cord("v0")).result(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(kvstore::Write(store, "k1", absl::Cord("v1"),
                             {StorageGeneration::NoValue()})
                  .result(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_TRUE(all_deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_F(GcsMockTest, AtomicZonalWriteQueryDestinationRetriesAndPropagatesError) {
  // Tests that when MoveObject falls back to QueryDestination (GetObject),
  // GetObject retries transient UNAVAILABLE errors with its own retry budget
  // (omitting x-goog-gcs-idempotency-token) and propagates PERMISSION_DENIED,
  // FAILED_PRECONDITION, and unconditional NOT_FOUND errors while cleaning up
  // the temporary object.
  std::string captured_write_id;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(4)
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        std::string temp_name;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            const auto& resource = req.write_object_spec().resource();
            temp_name = resource.name();
            auto it = resource.metadata().find("tensorstore-write-id");
            if (it != resource.metadata().end()) {
              captured_write_id = it->second;
            }
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

  ::testing::Sequence s;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(Return(
          grpc::Status(grpc::StatusCode::UNAVAILABLE, "transient move error")));
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          Return(grpc::Status(grpc::StatusCode::NOT_FOUND, "source missing")));
  EXPECT_CALL(mock(), GetObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          [](grpc::ServerContext* context, const GetObjectRequest*, auto*) {
            EXPECT_EQ(GetIdempotencyToken(context), "");
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "transient get error");
          });
  EXPECT_CALL(mock(), GetObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](grpc::ServerContext* context, const GetObjectRequest* req,
                    auto* resp) {
        EXPECT_EQ(GetIdempotencyToken(context), "");
        resp->set_name(req->object());
        resp->set_bucket(req->bucket());
        resp->set_generation(88);
        (*resp->mutable_metadata())["tensorstore-write-id"] = captured_write_id;
        return grpc::Status::OK;
      });

  for (auto err_code :
       {grpc::StatusCode::PERMISSION_DENIED,
        grpc::StatusCode::FAILED_PRECONDITION, grpc::StatusCode::NOT_FOUND}) {
    EXPECT_CALL(mock(), MoveObject(_, _, _))
        .InSequence(s)
        .WillOnce(Return(grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                      "transient move error")));
    EXPECT_CALL(mock(), MoveObject(_, _, _))
        .InSequence(s)
        .WillOnce(Return(
            grpc::Status(grpc::StatusCode::NOT_FOUND, "source missing")));
    EXPECT_CALL(mock(), GetObject(_, _, _))
        .InSequence(s)
        .WillOnce(Return(grpc::Status(err_code, "get failed")));
  }

  std::atomic<int> delete_count = 0;
  absl::Notification all_deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(3)
      .WillRepeatedly([&](auto*, const DeleteObjectRequest*, auto*) {
        if (delete_count.fetch_add(1) + 1 == 3) {
          all_deleted.Notify();
        }
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res1, kvstore::Write(store, "k0", absl::Cord("v0")).result());
  EXPECT_EQ(res1.generation, StorageGeneration::FromUint64(88));

  EXPECT_THAT(kvstore::Write(store, "k1", absl::Cord("v1")).result(),
              StatusIs(absl::StatusCode::kPermissionDenied));
  EXPECT_THAT(kvstore::Write(store, "k2", absl::Cord("v2"),
                             {StorageGeneration::FromUint64(10)})
                  .result(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(kvstore::Write(store, "k3", absl::Cord("v3")).result(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_TRUE(all_deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_F(GcsMockTest,
       AtomicZonalWriteZeroGenerationFromMoveOrGetObjectReturnsInternalError) {
  // Tests that both FinishMove and FinishQueryDestination reject an OK
  // response carrying generation 0 with an INTERNAL error and clean up the
  // temporary object.
  std::string captured_write_id;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .Times(2)
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        std::string temp_name;
        while (stream->Read(&req)) {
          if (req.has_write_object_spec()) {
            const auto& resource = req.write_object_spec().resource();
            temp_name = resource.name();
            auto it = resource.metadata().find("tensorstore-write-id");
            if (it != resource.metadata().end()) {
              captured_write_id = it->second;
            }
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

  ::testing::Sequence s;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce([](auto*, const MoveObjectRequest* req, auto* resp) {
        resp->set_name(req->destination_object());
        resp->set_bucket(req->bucket());
        resp->set_generation(0);
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(Return(
          grpc::Status(grpc::StatusCode::UNAVAILABLE, "transient move error")));
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .InSequence(s)
      .WillOnce(
          Return(grpc::Status(grpc::StatusCode::NOT_FOUND, "source missing")));
  EXPECT_CALL(mock(), GetObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](auto*, const GetObjectRequest* req, auto* resp) {
        resp->set_name(req->object());
        resp->set_bucket(req->bucket());
        resp->set_generation(0);
        (*resp->mutable_metadata())["tensorstore-write-id"] = captured_write_id;
        return grpc::Status::OK;
      });

  std::atomic<int> delete_count = 0;
  absl::Notification all_deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(2)
      .WillRepeatedly([&](auto*, const DeleteObjectRequest*, auto*) {
        if (delete_count.fetch_add(1) + 1 == 2) {
          all_deleted.Notify();
        }
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  EXPECT_THAT(kvstore::Write(store, "k0", absl::Cord("v0")).result(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("MoveObject completed with zero generation")));
  EXPECT_THAT(kvstore::Write(store, "k1", absl::Cord("v1")).result(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("GetObject completed with zero generation")));
  EXPECT_TRUE(all_deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_F(GcsMockTest, BidiWriteStopsFollowingRedirectsAfterMaxRedirects) {
  // Tests that BidiWriteTask follows at most kMaxRedirects (10) redirects
  // before falling through to normal status handling and terminating with
  // ABORTED.
  std::atomic<int> call_count = 0;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillRepeatedly([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        stream->Read(&req);
        int idx = call_count.fetch_add(1);
        google::storage::v2::BidiWriteObjectRedirectedError details;
        details.set_routing_token(absl::StrFormat("tok-%d", idx));
        google::rpc::Status rpc_status;
        rpc_status.set_code(static_cast<int>(grpc::StatusCode::ABORTED));
        rpc_status.set_message("redirect loop");
        rpc_status.add_details()->PackFrom(details);
        return grpc::Status(grpc::StatusCode::ABORTED, "redirect loop",
                            rpc_status.SerializeAsString());
      });

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {{"driver", "gcs_grpc"},
           {"endpoint", mock_service_.server_address()},
           {"bucket", "bucket"},
           {"experimental_mode", "standard"},
           {"timeout", "250ms"},
           {"context", {{"gcs_request_retries", {{"max_retries", 1}}}}}})
          .result());

  EXPECT_THAT(kvstore::Write(store, "key", absl::Cord("data")).result(),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("redirect loop")));
  EXPECT_EQ(call_count.load(), 11);
}

TEST_F(GcsMockTest, AtomicZonalWriteCancelledDuringMoveCleansTemp) {
  // Tests that cancelling a zonal write while MoveObject is in flight cancels
  // the move RPC and deletes the temporary object generation.
  std::string captured_temp_name;
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

  absl::Notification move_started;
  absl::Notification release_move;
  EXPECT_CALL(mock(), MoveObject(_, _, _))
      .WillOnce([&](auto*, const MoveObjectRequest*, auto*) {
        move_started.Notify();
        release_move.WaitForNotification();
        return grpc::Status(grpc::StatusCode::CANCELLED, "cancelled");
      });

  absl::Notification deleted;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .WillOnce([&](auto*, const DeleteObjectRequest* req, auto*) {
        EXPECT_EQ(req->object(), captured_temp_name);
        EXPECT_EQ(req->if_generation_match(), 42);
        deleted.Notify();
        return grpc::Status::OK;
      });

  auto store = OpenRapidStore();
  {
    auto write_future = kvstore::Write(store, "target", absl::Cord("data"));
    ASSERT_TRUE(move_started.WaitForNotificationWithTimeout(absl::Seconds(10)));
  }
  release_move.Notify();
  EXPECT_TRUE(deleted.WaitForNotificationWithTimeout(absl::Seconds(10)));
}

TEST_F(GcsMockTest, GrpcCallContextRetriesFirstUnauthenticatedOnlyOnce) {
  // Tests that GrpcCallContext::MaybeRetry retries UNAUTHENTICATED once
  // immediately on attempt 0 to mask credential refresh races, and stops if
  // attempt 1 also returns UNAUTHENTICATED.
  std::atomic<int> call_count = 0;
  ::testing::Sequence s;
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](auto*, const DeleteObjectRequest*, auto*) {
        call_count.fetch_add(1);
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED,
                            "expired credentials");
      });
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](auto*, const DeleteObjectRequest*, auto*) {
        call_count.fetch_add(1);
        return grpc::Status::OK;
      });

  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](auto*, const DeleteObjectRequest*, auto*) {
        call_count.fetch_add(1);
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED,
                            "invalid credentials");
      });
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .InSequence(s)
      .WillOnce([&](auto*, const DeleteObjectRequest*, auto*) {
        call_count.fetch_add(1);
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED,
                            "invalid credentials");
      });

  auto store = OpenModeStore("standard");
  TENSORSTORE_ASSERT_OK(kvstore::Delete(store, "k0").result());
  EXPECT_EQ(call_count.load(), 2);

  EXPECT_THAT(kvstore::Delete(store, "k1").result(),
              StatusIs(absl::StatusCode::kUnauthenticated,
                       HasSubstr("invalid credentials")));
  EXPECT_EQ(call_count.load(), 4);
}

TEST_F(GcsMockTest, NonAtomicZonalWriteDirectWithoutMove) {
  EXPECT_CALL(mock(), MoveObject(_, _, _)).Times(0);
  EXPECT_CALL(mock(), DeleteObject(_, _, _)).Times(0);
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        EXPECT_TRUE(req.has_write_object_spec());
        EXPECT_EQ(req.write_object_spec().resource().name(), "target_key");
        EXPECT_TRUE(req.write_object_spec().appendable());
        EXPECT_EQ(req.write_object_spec().object_size(), 4);
        EXPECT_TRUE(req.state_lookup());
        EXPECT_FALSE(req.finish_write());

        BidiWriteObjectResponse state_resp;
        state_resp.mutable_resource()->set_generation(42);
        stream->Write(state_resp);

        EXPECT_TRUE(stream->Read(&req));
        EXPECT_EQ(req.checksummed_data().content(), "data");
        EXPECT_TRUE(req.finish_write());

        BidiWriteObjectResponse final_resp;
        auto* res = final_resp.mutable_resource();
        res->set_name("target_key");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(42);
        stream->Write(final_resp);
        return grpc::Status::OK;
      });

  auto store = OpenModeStore("non_atomic");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto gen,
      kvstore::Write(store, "target_key", absl::Cord("data")).result());
  EXPECT_EQ(gen.generation, StorageGeneration::FromUint64(42));
}

TEST_F(GcsMockTest, NonAtomicZonalWriteWithGenerationConditionAndRetry) {
  EXPECT_CALL(mock(), MoveObject(_, _, _)).Times(0);
  EXPECT_CALL(mock(), DeleteObject(_, _, _)).Times(0);
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        EXPECT_TRUE(req.has_write_object_spec());
        EXPECT_EQ(req.write_object_spec().resource().name(), "cond_key");
        EXPECT_TRUE(req.write_object_spec().appendable());
        EXPECT_TRUE(req.write_object_spec().has_if_generation_match());
        EXPECT_EQ(req.write_object_spec().if_generation_match(), 0);
        EXPECT_TRUE(req.state_lookup());

        BidiWriteObjectResponse state_resp;
        state_resp.mutable_resource()->set_generation(7);
        stream->Write(state_resp);

        EXPECT_TRUE(stream->Read(&req));
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "stream reset");
      })
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        EXPECT_TRUE(req.has_write_object_spec());
        EXPECT_EQ(req.write_object_spec().resource().name(), "cond_key");
        EXPECT_TRUE(req.write_object_spec().appendable());
        EXPECT_TRUE(req.write_object_spec().has_if_generation_match());
        // The retry replaces the partial generation 7 created by attempt 1.
        EXPECT_EQ(req.write_object_spec().if_generation_match(), 7);
        EXPECT_FALSE(req.state_lookup());
        EXPECT_TRUE(req.finish_write());

        BidiWriteObjectResponse final_resp;
        auto* res = final_resp.mutable_resource();
        res->set_name("cond_key");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(8);
        stream->Write(final_resp);
        return grpc::Status::OK;
      });

  auto store = OpenModeStore("non_atomic");
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto gen, kvstore::Write(store, "cond_key", absl::Cord("data"),
                               {StorageGeneration::NoValue()})
                    .result());
  EXPECT_EQ(gen.generation, StorageGeneration::FromUint64(8));
}

TEST_F(GcsMockTest, AutoDetectRegionalProbeSettlesOnPreconditionMismatch) {
  // Tests that when the initial probe write on an unknown bucket is a
  // conditional write (if_equal) that receives FAILED_PRECONDITION, the
  // non-appendable acceptance settles is_zonal to false so subsequent writes
  // skip the 0-byte state_lookup probe.
  std::vector<bool> state_lookups;
  EXPECT_CALL(mock(), BidiWriteObject(_, _))
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        state_lookups.push_back(req.state_lookup());
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                            "condition not met");
      })
      .WillOnce([&](auto*, auto* stream) -> grpc::Status {
        BidiWriteObjectRequest req;
        EXPECT_TRUE(stream->Read(&req));
        state_lookups.push_back(req.state_lookup());
        EXPECT_FALSE(req.state_lookup());
        EXPECT_TRUE(req.finish_write());
        BidiWriteObjectResponse resp;
        auto* res = resp.mutable_resource();
        res->set_name("k1");
        res->set_bucket("projects/_/buckets/bucket");
        res->set_generation(55);
        stream->Write(resp);
        return grpc::Status::OK;
      });

  auto store = OpenStore();
  EXPECT_EQ(TestGetIsZonal(store), std::nullopt);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res0, kvstore::Write(store, "k0", absl::Cord("v0"),
                                {StorageGeneration::FromUint64(10)})
                     .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(res0.generation));
  EXPECT_EQ(TestGetIsZonal(store), false);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto res1, kvstore::Write(store, "k1", absl::Cord("v1")).result());
  EXPECT_EQ(res1.generation, StorageGeneration::FromUint64(55));
  EXPECT_THAT(state_lookups, testing::ElementsAre(true, false));
}

TEST_F(GcsMockTest, UnconditionalDeleteFailedPreconditionPropagatesError) {
  // Tests that an unconditional Delete propagates FAILED_PRECONDITION as an
  // error (e.g. retention policy / legal hold), while a conditional Delete
  // maps FAILED_PRECONDITION to StorageGeneration::Unknown().
  EXPECT_CALL(mock(), DeleteObject(_, _, _))
      .Times(2)
      .WillRepeatedly(Return(grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                          "Object is under active hold")));

  auto store = OpenModeStore("standard");
  EXPECT_THAT(kvstore::Delete(store, "k0").result(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("active hold")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto cond_res,
      kvstore::Delete(store, "k1", {StorageGeneration::FromUint64(10)})
          .result());
  EXPECT_TRUE(StorageGeneration::IsUnknown(cond_res.generation));
}

}  // namespace
