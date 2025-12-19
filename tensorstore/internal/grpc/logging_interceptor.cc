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

#include <map>
#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
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

ABSL_CONST_INIT internal_log::VerboseFlag verbose_logging("grpc_channel");

// Implements detailed logging for gRPC calls to GCS.
class LoggingInterceptor : public Interceptor {
 public:
  LoggingInterceptor(const ClientRpcInfo* info) : info_(info) {}

  std::string_view method_name() const { return info_->method(); }

  void Intercept(InterceptorBatchMethods* methods) override {
    if (!verbose_logging) {
      methods->Proceed();
      return;
    }

    std::string log;

    std::multimap<std::string, std::string>* metadata = nullptr;
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_INITIAL_METADATA)) {
      metadata = methods->GetSendInitialMetadata();
    }
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_STATUS)) {
      metadata = methods->GetSendTrailingMetadata();
    }
    if (metadata != nullptr) {
      absl::StrAppend(&log, "Metadata:");
      for (const auto& [key, value] : *metadata) {
        absl::StrAppend(&log, " ", key, ": ", value);
      }
    }

    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_MESSAGE)) {
      auto* message =
          static_cast<const google::protobuf::Message*>(methods->GetSendMessage());
      if (verbose_logging.Level(2) && message != nullptr) {
        absl::StrAppend(&log, log.empty() ? "" : "\n",
                        "Request: ", ConciseDebugString(*message));
      }
    }
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::POST_RECV_MESSAGE)) {
      auto* message =
          static_cast<const google::protobuf::Message*>(methods->GetRecvMessage());
      if (verbose_logging.Level(2) && message != nullptr) {
        absl::StrAppend(&log, log.empty() ? "" : "\n",
                        "Response: ", ConciseDebugString(*message));
      }
    }

    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_STATUS)) {
      if (auto status = methods->GetSendStatus(); !status.ok()) {
        absl::StrAppend(&log, log.empty() ? "" : "\n", "Send Status: ",
                        internal::GrpcStatusToAbslStatus(status));
      }
    }

    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::POST_RECV_STATUS)) {
      if (auto* status = methods->GetRecvStatus(); status != nullptr) {
        absl::StrAppend(&log, log.empty() ? "" : "\n", "Recv Status: ",
                        internal::GrpcStatusToAbslStatus(*status));
      }
    }
    ABSL_LOG_IF(INFO, !log.empty()) << method_name() << " " << log;
    methods->Proceed();
  }

 private:
  const ClientRpcInfo* info_;
};

}  // namespace

// Creates a logging interceptor.
Interceptor* LoggingInterceptorFactory::CreateClientInterceptor(
    ClientRpcInfo* info) {
  return new LoggingInterceptor(info);
}

}  // namespace internal_grpc
}  // namespace tensorstore
