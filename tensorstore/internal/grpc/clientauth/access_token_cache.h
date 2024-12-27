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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_CACHE_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_CACHE_H_

#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_grpc {

/// Cache asynchronously created access tokens.
///
/// This is a helper class to implement service account impersonation for gRPC.
/// Service account impersonation is implemented by querying the IAM Credentials
/// service, which returns a access token (an opaque string) when the
/// impersonation is allowed. These tokens can be cached, so the library does
/// not need to fetch the access token on each RPC.
///
/// Because we want to support asynchronous RPCs in the libraries, we need to
/// also fetch these access tokens asynchronously, or we would be blocking the
/// application while fetching the token.
///
/// Splitting this functionality to a separate class (instead of the
/// GrpcAuthenticationStrategy for service account impersonation) makes for
/// easier testing.
class AccessTokenCache : public std::enable_shared_from_this<AccessTokenCache> {
 public:
  using AccessToken = ::tensorstore::internal_grpc::AccessToken;
  using AccessTokenSource = std::function<Future<AccessToken>()>;

  static std::shared_ptr<AccessTokenCache> Create(AccessTokenSource source);

  Result<AccessToken> GetAccessToken(absl::Time now = absl::Now())
      ABSL_LOCKS_EXCLUDED(mu_);
  Future<AccessToken> AsyncGetAccessToken(absl::Time now = absl::Now())
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  using WaiterType = Future<AccessToken>;
  explicit AccessTokenCache(AccessTokenSource source);

  Future<AccessToken> StartRefresh() ABSL_UNLOCK_FUNCTION(mu_);
  void OnRefresh(Result<AccessToken>);

  std::weak_ptr<AccessTokenCache> WeakFromThis() {
    return std::weak_ptr<AccessTokenCache>(shared_from_this());
  }

  absl::Mutex mu_;
  AccessToken token_;
  Future<AccessToken> pending_;
  AccessTokenSource source_;
};

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_CACHE_H_
