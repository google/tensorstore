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

#ifndef TENSORSTORE_KVSTORE_TSGRPC_MOCK_SERVICE_H_
#define TENSORSTORE_KVSTORE_TSGRPC_MOCK_SERVICE_H_

#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/grpc_mock.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.pb.h"

namespace tensorstore_grpc {

// Mock KvStoreService to be used with  MockGrpcServer:
//
// Example:
//
//  grpc_mocker::MockGrpcServer<MockKvStoreService> mock_service;
//  EXPECT_CALL(*mock_service_.service(), Read)
//      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));
//
class MockKvStoreService : public kvstore::grpc_gen::KvStoreService::Service {
 public:
  using ServiceType = ::tensorstore_grpc::kvstore::grpc_gen::KvStoreService;

  TENSORSTORE_GRPC_MOCK(Read, ::tensorstore_grpc::kvstore::ReadRequest,
                        ::tensorstore_grpc::kvstore::ReadResponse);
  TENSORSTORE_GRPC_MOCK(Write, ::tensorstore_grpc::kvstore::WriteRequest,
                        ::tensorstore_grpc::kvstore::WriteResponse);
  TENSORSTORE_GRPC_MOCK(Delete, ::tensorstore_grpc::kvstore::DeleteRequest,
                        ::tensorstore_grpc::kvstore::DeleteResponse);
  TENSORSTORE_GRPC_SERVER_STREAMING_MOCK(
      List, ::tensorstore_grpc::kvstore::ListRequest,
      ::tensorstore_grpc::kvstore::ListResponse);
};

}  // namespace tensorstore_grpc

#endif  // TENSORSTORE_KVSTORE_TSGRPC_MOCK_SERVICE_H_
