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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IMPERSONATE_SERVICE_ACCOUNT_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IMPERSONATE_SERVICE_ACCOUNT_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/internal/grpc/clientauth/access_token_cache.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_grpc {

class GrpcAsyncAccessTokenCache;

struct ImpersonateServiceAccountConfig {
  std::string target_service_account;
  absl::Duration lifetime = absl::ZeroDuration();
  std::vector<std::string> scopes;
  std::vector<std::string> delegates;
};

class GrpcImpersonateServiceAccount
    : public GrpcAuthenticationStrategy,
      public std::enable_shared_from_this<GrpcImpersonateServiceAccount> {
  struct private_t {
    explicit private_t() = default;
  };

 public:
  static std::shared_ptr<GrpcImpersonateServiceAccount> Create(
      const ImpersonateServiceAccountConfig& config, const CaInfo& ca_info,
      std::shared_ptr<GrpcAuthenticationStrategy> base_strategy);

  GrpcImpersonateServiceAccount(private_t, const CaInfo& ca_info,
                                std::shared_ptr<AccessTokenCache> cache);

  ~GrpcImpersonateServiceAccount() override;

  std::shared_ptr<grpc::ChannelCredentials> GetChannelCredentials(
      std::string_view endpoint,
      grpc::ChannelArguments& arguments) const override;

  bool RequiresConfigureContext() const override;

  Future<std::shared_ptr<grpc::ClientContext>> ConfigureContext(
      std::shared_ptr<grpc::ClientContext>) const override;

 private:
  std::shared_ptr<grpc::CallCredentials> UpdateCallCredentials(
      const std::string& token);

  Result<std::shared_ptr<grpc::ClientContext>> OnGetCallCredentials(
      std::shared_ptr<grpc::ClientContext> context, Result<AccessToken> result);

  std::weak_ptr<GrpcImpersonateServiceAccount> WeakFromThis() const {
    return const_cast<GrpcImpersonateServiceAccount*>(this)->shared_from_this();
  }

  std::shared_ptr<AccessTokenCache> cache_;
  mutable absl::Mutex mu_;
  std::string access_token_;
  std::shared_ptr<grpc::CallCredentials> credentials_;
  grpc::SslCredentialsOptions ssl_options_;
};

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_IMPERSONATE_SERVICE_ACCOUNT_H_
