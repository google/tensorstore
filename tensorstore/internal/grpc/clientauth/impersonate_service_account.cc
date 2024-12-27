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

#include "tensorstore/internal/grpc/clientauth/impersonate_service_account.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "google/iam/credentials/v1/common.pb.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/internal/grpc/clientauth/access_token_cache.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/iam_stub.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_grpc {

/* static */
std::shared_ptr<GrpcImpersonateServiceAccount>
GrpcImpersonateServiceAccount::Create(
    const ImpersonateServiceAccountConfig& config, const CaInfo& ca_info,
    std::shared_ptr<GrpcAuthenticationStrategy> base_strategy) {
  auto source = CreateIamCredentialsSource(
      base_strategy, /*endpoint=*/{}, config.target_service_account,
      config.lifetime, config.scopes, config.delegates);
  return std::make_shared<GrpcImpersonateServiceAccount>(
      private_t{}, ca_info, AccessTokenCache::Create(std::move(source)));
}

GrpcImpersonateServiceAccount::GrpcImpersonateServiceAccount(
    private_t, const CaInfo& ca_info, std::shared_ptr<AccessTokenCache> cache)
    : cache_(std::move(cache)) {
  auto cainfo = LoadCAInfo(ca_info);
  if (cainfo) ssl_options_.pem_root_certs = std::move(*cainfo);
}

GrpcImpersonateServiceAccount::~GrpcImpersonateServiceAccount() = default;

std::shared_ptr<grpc::ChannelCredentials>
GrpcImpersonateServiceAccount::GetChannelCredentials(
    std::string_view endpoint, grpc::ChannelArguments& arguments) const {
  return grpc::SslCredentials(ssl_options_);
}

bool GrpcImpersonateServiceAccount::RequiresConfigureContext() const {
  return true;
}

Future<std::shared_ptr<grpc::ClientContext>>
GrpcImpersonateServiceAccount::ConfigureContext(
    std::shared_ptr<grpc::ClientContext> context) const {
  struct Callback {
    std::weak_ptr<GrpcImpersonateServiceAccount> w;
    std::shared_ptr<grpc::ClientContext> context;

    Result<std::shared_ptr<grpc::ClientContext>> operator()(
        Result<AccessToken> f) {
      auto self = w.lock();
      if (!self) {
        return absl::UnknownError(
            "lost reference to GrpcImpersonateServiceAccount");
      }
      return self->OnGetCallCredentials(std::move(context), f);
    }
  };
  return MapFuture(InlineExecutor{},
                   Callback{WeakFromThis(), std::move(context)},
                   cache_->AsyncGetAccessToken());
}

std::shared_ptr<grpc::CallCredentials>
GrpcImpersonateServiceAccount::UpdateCallCredentials(const std::string& token) {
  absl::MutexLock lock(&mu_);
  if (access_token_ != token) {
    access_token_ = token;
    credentials_ = grpc::AccessTokenCredentials(token);
  }
  return credentials_;
}

Result<std::shared_ptr<grpc::ClientContext>>
GrpcImpersonateServiceAccount::OnGetCallCredentials(
    std::shared_ptr<grpc::ClientContext> context, Result<AccessToken> result) {
  if (!result.ok()) return std::move(result).status();
  context->set_credentials(UpdateCallCredentials(result->token));
  return context;
}

}  // namespace internal_grpc
}  // namespace tensorstore
