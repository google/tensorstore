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

#include "tensorstore/kvstore/gcs_http/shared_auth_provider.h"

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/internal/oauth2/google_auth_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_gcs_http {
namespace {

using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_oauth2::AuthProvider;

struct SharedAuthProvider {
  absl::Mutex mu_;
  std::weak_ptr<HttpTransport> transport_ ABSL_GUARDED_BY(mu_);
  Result<std::shared_ptr<AuthProvider>> auth_provider_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

Result<std::shared_ptr<AuthProvider>> GetSharedGoogleAuthProvider(
    std::shared_ptr<HttpTransport> transport) {
  static auto* g = new SharedAuthProvider();

  absl::MutexLock l(&g->mu_);
  // Return the cached auth provider if the transport is the same; otherwise
  // update the transport and create a new auth provider.
  if (std::shared_ptr<HttpTransport> cached_transport = g->transport_.lock()) {
    if (transport == cached_transport) return g->auth_provider_;
  }
  g->transport_ = transport;
  g->auth_provider_ = internal_oauth2::GetGoogleAuthProvider(transport);
  return g->auth_provider_;
}

}  // namespace internal_kvstore_gcs_http
}  // namespace tensorstore
