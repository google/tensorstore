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

#ifndef TENSORSTORE_INTERNAL_GRPC_GRPC_MOCK_H_
#define TENSORSTORE_INTERNAL_GRPC_GRPC_MOCK_H_

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include "absl/log/absl_check.h"
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/grpcpp.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party

namespace tensorstore {
namespace grpc_mocker {

// Mock gRPC server implementation helper. gRPC doesn't provide a native
// mocking mechanism that works with the async() interface, so this strategy
// allows creating a mock instance to achieve that.
//
// Example:
//
//  class MockEchoService : public EchoService::Service {
//    public:
//     using ServiceType = EchoService
//     TENSORSTORE_GRPC_MOCK(Echo, EchoProto, EchoProto);
//  };
//
//  MockGrpcServer<MockEchoService> mock_service;
//
//  EXPECT_CALL(*mock_service.service(),
//              Echo(testing::_, testing::_, testing::_))
//      .WillRepeatedly(testing::Invoke([](auto, auto* request, auto* response)
//      {
//          *request = *response;
//          return regrpc::Status::OK;
//       )));
//
//  EchoClient client(mock_service.stub());
//
template <typename ServiceImpl>
class MockGrpcServer {
 public:
  using ServiceType = typename ServiceImpl::ServiceType;
  using ServiceStub = typename ServiceType::Stub;

  // Initializes the gRPC server with local credentials.
  MockGrpcServer() : server_(BuildServer()) {
    ABSL_CHECK(server_);
    stub_ = NewStub();
  }

  ~MockGrpcServer() {
    server_->Shutdown();
    server_->Wait();
  }

  // Listening server address.
  const std::string& server_address() const { return server_address_; }

  // Service accessor; use to sett EXPECT_CALL expectations.
  ServiceImpl* service() { return &service_; }

  // Provides access to a stub which is connected to the mock server using local
  // credentials. The pointer's lifetime is tied to this MockGrpcServer
  // instance.
  std::shared_ptr<ServiceStub> stub() { return stub_; }

  // Creates a stub that communicates with the mock server using local
  // credentials.
  std::shared_ptr<ServiceStub> NewStub() {
    return ServiceType::NewStub(::grpc::CreateChannel(
        server_address_, ::grpc::experimental::LocalCredentials(LOCAL_TCP)));
  }

 private:
  // Setup the gRPC server.
  std::unique_ptr<::grpc::Server> BuildServer() {
    int port;
    auto server =
        grpc::ServerBuilder{}
            .AddListeningPort(
                "[::1]:0",
                grpc::experimental::LocalServerCredentials(LOCAL_TCP), &port)
            .RegisterService(&service_)
            .BuildAndStart();

    server_address_ = absl::StrFormat("[::1]:%d", port);
    return server;
  }

  ServiceImpl service_;

  // Local server addrress.
  std::string server_address_;

  const std::unique_ptr<::grpc::Server> server_;
  std::shared_ptr<ServiceStub> stub_;
};

#define TENSORSTORE_GRPC_MOCK(method, request, response) \
  MOCK_METHOD(::grpc::Status, method,                    \
              (::grpc::ServerContext*, const request*, response*))

}  // namespace grpc_mocker
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_GRPC_MOCK_H_
