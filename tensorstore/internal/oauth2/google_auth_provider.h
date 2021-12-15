// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_PROVIDER_H_
#define TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_PROVIDER_H_

#include <functional>
#include <memory>

#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

/// \brief Return an AuthProvider for a Google environment.
///
/// The flow is:
///   1. Is the test env GOOGLE_AUTH_TOKEN_FOR_TESTING set?
///   2. What is in the local credentials file?
///     2a. check for GOOGLE_APPLICATION_CREDENTIALS
///     2b. check well known locations for application_default_credentials.json
///        ${CLOUDSDK_CONFIG,HOME}/.config/gcloud/...
///   3. Are we running on GCE?
///   4. Otherwise fail.
///
Result<std::unique_ptr<AuthProvider>> GetGoogleAuthProvider(
    std::shared_ptr<internal_http::HttpTransport> transport =
        internal_http::GetDefaultHttpTransport());

/// Returns a shared AuthProvider for Google Cloud credentials.
///
/// Repeated calls will return the same instance unless
/// `ResetSharedGoogleAuthProvider` is called.
Result<std::shared_ptr<AuthProvider>> GetSharedGoogleAuthProvider();

/// Ensures that the next call to `GetSharedGoogleAuthProvider` will return a
/// new instance rather than the previously-obtained auth provider.
///
/// This is primarily useful for unit tests where a mock HTTP transport is used.
void ResetSharedGoogleAuthProvider();

using GoogleAuthProvider =
    std::function<Result<std::unique_ptr<AuthProvider>>()>;
void RegisterGoogleAuthProvider(GoogleAuthProvider provider, int priority);

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_PROVIDER_H_
