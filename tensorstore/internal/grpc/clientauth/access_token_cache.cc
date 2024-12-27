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

#include "tensorstore/internal/grpc/clientauth/access_token_cache.h"

#include <memory>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/grpc/clientauth/access_token.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

using AccessTokenSource =
    ::tensorstore::internal_grpc::AccessTokenCache::AccessTokenSource;

namespace tensorstore {
namespace internal_grpc {
namespace {

constexpr auto kUseSlack = absl::Seconds(30);
constexpr auto kRefreshSlack = absl::Minutes(5);

}  // namespace

std::shared_ptr<AccessTokenCache> AccessTokenCache::Create(
    AccessTokenSource source) {
  return std::shared_ptr<AccessTokenCache>(
      new AccessTokenCache(std::move(source)));
}

AccessTokenCache::AccessTokenCache(AccessTokenSource source)
    : source_(std::move(source)) {}

Result<AccessToken> AccessTokenCache::GetAccessToken(absl::Time now) {
  return AsyncGetAccessToken(now).result();
}

Future<AccessToken> AccessTokenCache::AsyncGetAccessToken(absl::Time now) {
  mu_.Lock();
  if (now + kUseSlack > token_.expiration) {
    return StartRefresh();
  }
  auto tmp = token_;
  if (now + kRefreshSlack >= token_.expiration) {
    StartRefresh().IgnoreFuture();
  } else {
    mu_.Unlock();
  }
  return tmp;
}

Future<AccessToken> AccessTokenCache::StartRefresh() {
  mu_.AssertHeld();
  if (!pending_.null()) {
    mu_.Unlock();
    return pending_;
  }
  pending_ = source_();
  auto tmp = pending_;
  auto w = WeakFromThis();
  mu_.Unlock();

  pending_.ExecuteWhenReady([w](ReadyFuture<AccessToken> f) {
    if (auto self = w.lock()) self->OnRefresh(std::move(f).result());
  });
  return tmp;
}

void AccessTokenCache::OnRefresh(Result<AccessToken> f) {
  absl::MutexLock lk(&mu_);
  pending_ = {};
  if (f.status().ok()) {
    token_ = *std::move(f);
  }
}

}  // namespace internal_grpc
}  // namespace tensorstore
