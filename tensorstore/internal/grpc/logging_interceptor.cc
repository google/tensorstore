// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/grpc/logging_interceptor.h"

#include <string_view>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "grpcpp/support/client_interceptor.h"  // third_party
#include "grpcpp/support/interceptor.h"  // third_party
#include "google/protobuf/message.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/proto/proto_util.h"

using ::grpc::experimental::ClientRpcInfo;
using ::grpc::experimental::InterceptionHookPoints;
using ::grpc::experimental::Interceptor;
using ::grpc::experimental::InterceptorBatchMethods;

namespace tensorstore {
namespace internal_grpc {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag grpc_logging("grpc_channel");

// Implements detailed logging for gRPC calls to GCS.
class LoggingInterceptor : public Interceptor {
 public:
  LoggingInterceptor(const ClientRpcInfo* info) : info_(info) {}

  std::string_view method_name() const { return info_->method(); }

  void Intercept(InterceptorBatchMethods* methods) override {
    if (!grpc_logging) {
      methods->Proceed();
      return;
    }
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_MESSAGE)) {
      auto* message =
          static_cast<const google::protobuf::Message*>(methods->GetSendMessage());
      bool is_first_message = !started_;
      started_ = true;
      if (grpc_logging.Level(2)) {
        ABSL_LOG(INFO) << "Begin: " << method_name() << " "
                       << ConciseDebugString(*message);
      } else if (is_first_message) {
        ABSL_LOG(INFO) << "Begin: " << method_name();
      }
    }
    if (grpc_logging.Level(2) &&
        methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::POST_RECV_MESSAGE)) {
      ABSL_LOG(INFO) << method_name() << " "
                     << ConciseDebugString(*static_cast<const google::protobuf::Message*>(
                            methods->GetRecvMessage()));
    }
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::POST_RECV_STATUS)) {
      if (auto* status = methods->GetRecvStatus();
          status != nullptr && !status->ok()) {
        ABSL_LOG(INFO) << "Error: " << method_name() << " "
                       << internal::GrpcStatusToAbslStatus(*status);
      } else {
        ABSL_LOG(INFO) << "End: " << method_name();
      }
    }
    methods->Proceed();
  }

 private:
  const ClientRpcInfo* info_;
  bool started_ = false;
};

}  // namespace

// Creates a logging interceptor.
Interceptor* LoggingInterceptorFactory::CreateClientInterceptor(
    ClientRpcInfo* info) {
  return new LoggingInterceptor(info);
}

}  // namespace internal_grpc
}  // namespace tensorstore
