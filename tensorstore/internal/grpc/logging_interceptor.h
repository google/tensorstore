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

#ifndef TENSORSTORE_INTERNAL_GRPC_LOGGING_INTERCEPTOR_H_
#define TENSORSTORE_INTERNAL_GRPC_LOGGING_INTERCEPTOR_H_

#include "grpcpp/support/client_interceptor.h"  // third_party
#include "grpcpp/support/interceptor.h"  // third_party

namespace tensorstore {
namespace internal_grpc {

/// Creates a logging interceptor factory to log gRPC calls.
class LoggingInterceptorFactory
    : public grpc::experimental::ClientInterceptorFactoryInterface {
 public:
  grpc::experimental::Interceptor* CreateClientInterceptor(
      grpc::experimental::ClientRpcInfo* info) override;
};

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_LOGGING_INTERCEPTOR_H_
