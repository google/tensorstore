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

#include <cassert>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpc/support/log.h"
#include "grpcpp/completion_queue.h"  // third_party
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/grpcpp.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/server_context.h"  // third_party

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
// NOTE: Tests using the MockGrpcServer may expose a race condition in gRPC
// which leads to a crash related to grpc_completion_queue_next, pthrea_join,
// or CallbackAlternativeCQ.  This can happen when a callback handler releases
// the last reference to a gRPC channel / stub, which may trigger gRPC to
// shutdown an internal threadpool, which just happens to be the threadpool
// the callback uses to run.
// See: https://github.com/grpc/grpc/pull/32584
template <typename ServiceImpl>
class MockGrpcServer {
 public:
  using ServiceType = typename ServiceImpl::ServiceType;
  using ServiceStub = typename ServiceType::Stub;

  // Initializes the gRPC server with local credentials.
  explicit MockGrpcServer() {
    BuildServer();
    ABSL_CHECK(server_);
    stub_ = NewStub();
  }

  ~MockGrpcServer() {
    server_->Shutdown(absl::ToChronoTime(absl::Now()));
    server_->Wait();
  }

  // Listening server address.
  std::string server_address() const {
    return absl::StrFormat("localhost:%d", port_);
  }

  // Service accessor; use to set EXPECT_CALL expectations.
  ServiceImpl* service() { return &service_; }

  // Provides access to a stub which is connected to the mock server using local
  // credentials. The pointer's lifetime is tied to this MockGrpcServer
  // instance.
  std::shared_ptr<ServiceStub> stub() { return stub_; }

  // Creates a stub that communicates with the mock server using local
  // credentials.
  std::unique_ptr<ServiceStub> NewStub() {
    return ServiceType::NewStub(::grpc::CreateChannel(
        server_address(), ::grpc::experimental::LocalCredentials(LOCAL_TCP)));
  }

  void Shutdown(absl::Time deadline = absl::InfiniteFuture()) {
    server_->Shutdown(absl::ToChronoTime(deadline));
  }

 private:
  // Setup the gRPC server.
  void BuildServer() {
    gpr_set_log_verbosity(GPR_LOG_SEVERITY_INFO);

    assert(!server_);

    ::grpc::ServerBuilder builder;

    builder.SetSyncServerOption(::grpc::ServerBuilder::NUM_CQS, 2)
        .SetSyncServerOption(::grpc::ServerBuilder::MIN_POLLERS, 1)
        .SetSyncServerOption(::grpc::ServerBuilder::MAX_POLLERS, 2)
        .SetSyncServerOption(::grpc::ServerBuilder::CQ_TIMEOUT_MSEC, 10000);

    builder.AddListeningPort(
        "localhost:0", grpc::experimental::LocalServerCredentials(LOCAL_TCP),
        &port_);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
  }

  ServiceImpl service_;

  // Local server address.
  int port_ = 0;
  std::unique_ptr<::grpc::Server> server_;
  std::shared_ptr<ServiceStub> stub_;
};

#define TENSORSTORE_GRPC_MOCK(method, request, response) \
  MOCK_METHOD(::grpc::Status, method,                    \
              (::grpc::ServerContext*, const request*, response*))

#define TENSORSTORE_GRPC_CLIENT_STREAMING_MOCK(method, request, response) \
  MOCK_METHOD(                                                            \
      ::grpc::Status, method,                                             \
      (::grpc::ServerContext*, ::grpc::ServerReader<request>*, response*))

#define TENSORSTORE_GRPC_SERVER_STREAMING_MOCK(method, request, response) \
  MOCK_METHOD(::grpc::Status, method,                                     \
              (::grpc::ServerContext*, const request*,                    \
               ::grpc::ServerWriter<response>*))

}  // namespace grpc_mocker
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_GRPC_MOCK_H_
