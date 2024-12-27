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

#include "tensorstore/internal/grpc/clientauth/iam_stub.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "google/iam/credentials/v1/common.pb.h"
#include "google/iam/credentials/v1/common.pb.h"
#include "google/iam/credentials/v1/iamcredentials.grpc.pb.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/grpcpp.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/create_channel.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

using ::google::iam::credentials::v1::GenerateAccessTokenRequest;
using ::google::iam::credentials::v1::GenerateAccessTokenResponse;
using ::google::iam::credentials::v1::IAMCredentials;

namespace tensorstore {
namespace internal_grpc {
namespace {

constexpr auto kIamCredentialsEndpoint = "iamcredentials.googleapis.com";
constexpr auto kDefaultScope = "https://www.googleapis.com/auth/cloud-platform";
constexpr auto kDefaultTokenLifetime = absl::Hours(1);

class DefaultIamCredentialsStub : public IamCredentialsStub {
 public:
  DefaultIamCredentialsStub(
      std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy,
      std::shared_ptr<IAMCredentials::StubInterface> stub)
      : auth_strategy_(std::move(auth_strategy)), stub_(std::move(stub)) {}

  ~DefaultIamCredentialsStub() override = default;

  Future<::google::iam::credentials::v1::GenerateAccessTokenResponse>
  AsyncGenerateAccessToken(
      std::shared_ptr<grpc::ClientContext> context,
      const ::google::iam::credentials::v1::GenerateAccessTokenRequest& request)
      override;

 private:
  std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy_;
  std::shared_ptr<IAMCredentials::StubInterface> stub_;
};

Future<::google::iam::credentials::v1::GenerateAccessTokenResponse>
DefaultIamCredentialsStub::AsyncGenerateAccessToken(
    std::shared_ptr<grpc::ClientContext> context,
    const ::google::iam::credentials::v1::GenerateAccessTokenRequest& request) {
  auto pair = PromiseFuturePair<GenerateAccessTokenResponse>::Make();
  LinkValue(
      [stub = stub_, request = request](
          Promise<GenerateAccessTokenResponse> p,
          ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
        auto context = std::move(f).value();
        context->AddMetadata(
            "x-goog-request-params",
            absl::StrCat("name=",
                         internal::PercentEncodeUriPath(request.name())));

        auto response = std::make_shared<GenerateAccessTokenResponse>();

        stub->async()->GenerateAccessToken(
            context.get(), &request, response.get(),
            [promise = std::move(p), context = std::move(context),
             response = std::move(response)](::grpc::Status s) {
              if (!promise.result_needed()) return;
              if (auto status = internal::GrpcStatusToAbslStatus(s);
                  !status.ok()) {
                promise.SetResult(status);
              } else {
                promise.SetResult(*std::move(response));
              }
            });
      },
      std::move(pair.promise),
      auth_strategy_->ConfigureContext(std::move(context)));

  return std::move(pair.future);
}

}  // namespace

std::shared_ptr<IamCredentialsStub> CreateIamCredentialsStub(
    std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy,
    std::string_view endpoint) {
  if (endpoint.empty()) {
    endpoint = kIamCredentialsEndpoint;
  }

  grpc::ChannelArguments args;
  auto channel = CreateChannel(*auth_strategy, std::string(endpoint), args);
  if (!channel) {
    return nullptr;
  }
  return std::make_shared<DefaultIamCredentialsStub>(
      auth_strategy, IAMCredentials::NewStub(channel));
}

std::function<Future<AccessToken>()> CreateIamCredentialsSource(
    std::shared_ptr<GrpcAuthenticationStrategy> auth_strategy,
    std::string_view endpoint, std::string_view target_service_account,
    absl::Duration lifetime, tensorstore::span<const std::string> scopes,
    tensorstore::span<const std::string> delegates) {
  auto stub = CreateIamCredentialsStub(auth_strategy, endpoint);

  GenerateAccessTokenRequest request;
  request.set_name(
      absl::StrCat("projects/-/serviceAccounts/", target_service_account));
  *request.mutable_delegates() = {delegates.begin(), delegates.end()};
  if (scopes.empty()) {
    request.add_scope(kDefaultScope);
  } else {
    *request.mutable_scope() = {scopes.begin(), scopes.end()};
  }
  request.mutable_lifetime()->set_seconds(absl::ToInt64Seconds(
      lifetime == absl::ZeroDuration() ? kDefaultTokenLifetime : lifetime));

  return [stub, request]() -> Future<AccessToken> {
    return MapFutureValue(
        InlineExecutor{},
        [](GenerateAccessTokenResponse& response) -> Result<AccessToken> {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto expiration,
              internal::ProtoToAbslTime(response.expire_time()));
          return AccessToken{*std::move(response.mutable_access_token()),
                             expiration};
        },
        stub->AsyncGenerateAccessToken(std::make_shared<grpc::ClientContext>(),
                                       request));
  };
}

}  // namespace internal_grpc
}  // namespace tensorstore
