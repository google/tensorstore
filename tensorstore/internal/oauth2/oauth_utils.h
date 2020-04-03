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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_OAUTH_UTILS_H_
#define TENSORSTORE_INTERNAL_OAUTH2_OAUTH_UTILS_H_

#include <stdint.h>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

/// Given a private key and a payload
Result<std::string> SignWithRSA256(absl::string_view private_key,
                                   absl::string_view to_sign);

/// Encodes a claim for a JSON web token (JWT) to make an OAuth request.
std::string BuildJWTHeader(absl::string_view key_id);

std::string BuildJWTClaimBody(absl::string_view client_email,
                              absl::string_view scope,  //
                              absl::string_view audience, absl::Time now,
                              int64_t lifetime = 3600);

/// Builds a request body for a JWT Claim.
Result<std::string> BuildSignedJWTRequest(absl::string_view private_key,
                                          absl::string_view header,
                                          absl::string_view body);

/// A parsed GoogleServiceAccountCredentials object.
struct GoogleServiceAccountCredentials {
  std::string private_key_id;
  std::string private_key;
  std::string token_uri;
  std::string client_email;
};

Result<GoogleServiceAccountCredentials>
ParseGoogleServiceAccountCredentialsImpl(const ::nlohmann::json& credentials);

Result<GoogleServiceAccountCredentials> ParseGoogleServiceAccountCredentials(
    absl::string_view source);

template <typename T>
absl::enable_if_t<std::is_same<T, ::nlohmann::json>::value,
                  Result<GoogleServiceAccountCredentials>>
ParseGoogleServiceAccountCredentials(const T& json) {
  return ParseGoogleServiceAccountCredentialsImpl(json);
}

/// A parsed RefreshToken object.
struct RefreshToken {
  std::string client_id;
  std::string client_secret;
  std::string refresh_token;
};
Result<RefreshToken> ParseRefreshTokenImpl(const ::nlohmann::json& credentials);

Result<RefreshToken> ParseRefreshToken(absl::string_view source);

template <typename T>
absl::enable_if_t<std::is_same<T, ::nlohmann::json>::value,
                  Result<RefreshToken>>
ParseRefreshToken(const T& json) {
  return ParseRefreshTokenImpl(json);
}

/// A parsed OAuthResponse object.
struct OAuthResponse {
  int64_t expires_in;
  std::string token_type;
  std::string access_token;
};
Result<OAuthResponse> ParseOAuthResponseImpl(
    const ::nlohmann::json& credentials);

Result<OAuthResponse> ParseOAuthResponse(absl::string_view source);

template <typename T>
absl::enable_if_t<std::is_same<T, ::nlohmann::json>::value,
                  Result<OAuthResponse>>
ParseOAuthResponse(const T& json) {
  return ParseOAuthResponseImpl(json);
}

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_OAUTH_UTILS_H_
