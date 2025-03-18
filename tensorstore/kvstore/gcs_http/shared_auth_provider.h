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

#ifndef TENSORSTORE_KVSTORE_GCS_HTTP_SHARED_AUTH_PROVIDER_H_
#define TENSORSTORE_KVSTORE_GCS_HTTP_SHARED_AUTH_PROVIDER_H_

#include <memory>

#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_gcs_http {

/// Returns a shared AuthProvider for Google Cloud credentials.
///
/// Caches the AuthProvider for the given HttpTransport; if the HttpTransport
/// changes, a new AuthProvider instance will be cached and returned.
Result<std::shared_ptr<internal_oauth2::AuthProvider>>
    GetSharedGoogleAuthProvider(std::shared_ptr<internal_http::HttpTransport>);

}  // namespace internal_kvstore_gcs_http
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_HTTP_SHARED_AUTH_PROVIDER_H_
