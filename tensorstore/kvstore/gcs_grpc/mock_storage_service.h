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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_MOCK_STORAGE_SERVICE_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_MOCK_STORAGE_SERVICE_H_

#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/grpc_mock.h"

namespace tensorstore_grpc {

// Mock Storage service to be used with MockGrpcServer:
//
// Example:
//
//  grpc_mocker::MockGrpcServer<MockStorage> mock_service;
//  EXPECT_CALL(*mock_service_.service(), ListObjects)
//      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));
//
class MockStorage : public ::google::storage::v2::Storage::Service {
 public:
  using ServiceType = ::google::storage::v2::Storage;

  TENSORSTORE_GRPC_MOCK(DeleteBucket,
                        ::google::storage::v2::DeleteBucketRequest,
                        ::google::protobuf::Empty);
  TENSORSTORE_GRPC_MOCK(GetBucket, ::google::storage::v2::GetBucketRequest,
                        ::google::storage::v2::Bucket);
  TENSORSTORE_GRPC_MOCK(CreateBucket,
                        ::google::storage::v2::CreateBucketRequest,
                        ::google::storage::v2::Bucket);
  TENSORSTORE_GRPC_MOCK(ListBuckets, ::google::storage::v2::ListBucketsRequest,
                        ::google::storage::v2::ListBucketsResponse);
  TENSORSTORE_GRPC_MOCK(LockBucketRetentionPolicy,
                        ::google::storage::v2::LockBucketRetentionPolicyRequest,
                        ::google::storage::v2::Bucket);
  TENSORSTORE_GRPC_MOCK(GetIamPolicy, ::google::iam::v1::GetIamPolicyRequest,
                        ::google::iam::v1::Policy);
  TENSORSTORE_GRPC_MOCK(SetIamPolicy, ::google::iam::v1::SetIamPolicyRequest,
                        ::google::iam::v1::Policy);
  TENSORSTORE_GRPC_MOCK(TestIamPermissions,
                        ::google::iam::v1::TestIamPermissionsRequest,
                        ::google::iam::v1::TestIamPermissionsResponse);
  TENSORSTORE_GRPC_MOCK(UpdateBucket,
                        ::google::storage::v2::UpdateBucketRequest,
                        ::google::storage::v2::Bucket);
  TENSORSTORE_GRPC_MOCK(DeleteNotificationConfig,
                        ::google::storage::v2::DeleteNotificationConfigRequest,
                        ::google::protobuf::Empty);
  TENSORSTORE_GRPC_MOCK(GetNotificationConfig,
                        ::google::storage::v2::GetNotificationConfigRequest,
                        ::google::storage::v2::NotificationConfig);
  TENSORSTORE_GRPC_MOCK(CreateNotificationConfig,
                        ::google::storage::v2::CreateNotificationConfigRequest,
                        ::google::storage::v2::NotificationConfig);
  TENSORSTORE_GRPC_MOCK(ListNotificationConfigs,
                        ::google::storage::v2::ListNotificationConfigsRequest,
                        ::google::storage::v2::ListNotificationConfigsResponse);
  TENSORSTORE_GRPC_MOCK(ComposeObject,
                        ::google::storage::v2::ComposeObjectRequest,
                        ::google::storage::v2::Object);
  TENSORSTORE_GRPC_MOCK(DeleteObject,
                        ::google::storage::v2::DeleteObjectRequest,
                        ::google::protobuf::Empty);
  TENSORSTORE_GRPC_MOCK(CancelResumableWrite,
                        ::google::storage::v2::CancelResumableWriteRequest,
                        ::google::storage::v2::CancelResumableWriteResponse);
  TENSORSTORE_GRPC_MOCK(GetObject, ::google::storage::v2::GetObjectRequest,
                        ::google::storage::v2::Object);
  TENSORSTORE_GRPC_SERVER_STREAMING_MOCK(
      ReadObject, ::google::storage::v2::ReadObjectRequest,
      ::google::storage::v2::ReadObjectResponse);
  TENSORSTORE_GRPC_MOCK(UpdateObject,
                        ::google::storage::v2::UpdateObjectRequest,
                        ::google::storage::v2::Object);
  TENSORSTORE_GRPC_CLIENT_STREAMING_MOCK(
      WriteObject, ::google::storage::v2::WriteObjectRequest,
      ::google::storage::v2::WriteObjectResponse);
  TENSORSTORE_GRPC_MOCK(ListObjects, ::google::storage::v2::ListObjectsRequest,
                        ::google::storage::v2::ListObjectsResponse);
  TENSORSTORE_GRPC_MOCK(RewriteObject,
                        ::google::storage::v2::RewriteObjectRequest,
                        ::google::storage::v2::RewriteResponse);
  TENSORSTORE_GRPC_MOCK(StartResumableWrite,
                        ::google::storage::v2::StartResumableWriteRequest,
                        ::google::storage::v2::StartResumableWriteResponse);
  TENSORSTORE_GRPC_MOCK(QueryWriteStatus,
                        ::google::storage::v2::QueryWriteStatusRequest,
                        ::google::storage::v2::QueryWriteStatusResponse);
  TENSORSTORE_GRPC_MOCK(GetServiceAccount,
                        ::google::storage::v2::GetServiceAccountRequest,
                        ::google::storage::v2::ServiceAccount);
  TENSORSTORE_GRPC_MOCK(CreateHmacKey,
                        ::google::storage::v2::CreateHmacKeyRequest,
                        ::google::storage::v2::CreateHmacKeyResponse);
  TENSORSTORE_GRPC_MOCK(DeleteHmacKey,
                        ::google::storage::v2::DeleteHmacKeyRequest,
                        ::google::protobuf::Empty);
  TENSORSTORE_GRPC_MOCK(GetHmacKey, ::google::storage::v2::GetHmacKeyRequest,
                        ::google::storage::v2::HmacKeyMetadata);
  TENSORSTORE_GRPC_MOCK(ListHmacKeys,
                        ::google::storage::v2::ListHmacKeysRequest,
                        ::google::storage::v2::ListHmacKeysResponse);
  TENSORSTORE_GRPC_MOCK(UpdateHmacKey,
                        ::google::storage::v2::UpdateHmacKeyRequest,
                        ::google::storage::v2::HmacKeyMetadata);
};

}  // namespace tensorstore_grpc

#endif  // TENSORSTORE_GOOGLE3_ONLY_GCS_GRPC_MOCK_STORAGE_SERVICE_H_
