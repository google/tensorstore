// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IAM_STUB_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IAM_STUB_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "google/iam/credentials/v1/common.pb.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/grpcpp.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_grpc {

class IamCredentialsStub {
 public:
  using GenerateAccessTokenResponse =
      ::google::iam::credentials::v1::GenerateAccessTokenResponse;
  using GenerateAccessTokenRequest =
      ::google::iam::credentials::v1::GenerateAccessTokenRequest;

  virtual ~IamCredentialsStub() = default;

  virtual Future<GenerateAccessTokenResponse> AsyncGenerateAccessToken(
      std::shared_ptr<grpc::ClientContext> context,
      const GenerateAccessTokenRequest& request) = 0;
};

std::shared_ptr<IamCredentialsStub> CreateIamCredentialsStub(
    std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy,
    std::string_view endpoint);

std::function<Future<AccessToken>()> CreateIamCredentialsSource(
    std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy,
    std::string_view endpoint, std::string_view target_service_account,
    absl::Duration lifetime, tensorstore::span<const std::string> scopes,
    tensorstore::span<const std::string> delegates);

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IAM_STUB_H_
