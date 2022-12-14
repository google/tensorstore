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

#include "tensorstore/internal/oauth2/oauth_utils.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/time/time.h"
#include <openssl/bio.h>     // IWYU pragma: keep
#include <openssl/evp.h>     // IWYU pragma: keep
#include <openssl/pem.h>     // IWYU pragma: keep
#include <openssl/rsa.h>     // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace jb = tensorstore::internal_json_binding;

namespace tensorstore {
namespace {

// The crypto algorithm to be used with OAuth.
constexpr char kCryptoAlgorithm[] = "RS256";

// The token type for the OAuth request.
constexpr char kJwtType[] = "JWT";

// The grant type for the OAuth request. Already URL-encoded for convenience.
constexpr char kGrantType[] =
    "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer";

}  // namespace
namespace internal_oauth2 {

Result<std::string> SignWithRSA256(std::string_view private_key,
                                   std::string_view to_sign) {
  if (private_key.empty()) {
    return absl::InternalError("No private key provided.");
  }

  const auto md = EVP_sha256();
  assert(md != nullptr);

  auto md_ctx = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
      EVP_MD_CTX_create(), &EVP_MD_CTX_free);
  assert(md_ctx != nullptr);

  auto pem_buffer = std::unique_ptr<BIO, decltype(&BIO_free)>(
      BIO_new_mem_buf(static_cast<const char*>(private_key.data()),
                      static_cast<int>(private_key.length())),
      &BIO_free);
  if (!pem_buffer) {
    return absl::InternalError("Could not create the PEM buffer.");
  }

  auto key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>(
      PEM_read_bio_PrivateKey(
          static_cast<BIO*>(pem_buffer.get()),
          nullptr,  // EVP_PKEY **x
          nullptr,  // pem_password_cb *cb -- a custom callback.
          // void *u -- this represents the password for the PEM (only
          // applicable for formats such as PKCS12 (.p12 files) that use
          // a password, which we don't currently support.
          nullptr),
      &EVP_PKEY_free);
  if (!key) {
    return absl::InternalError("Could not load the private key.");
  }

  if (EVP_DigestSignInit(md_ctx.get(), nullptr, md, nullptr, key.get()) != 1) {
    return absl::InternalError("DigestInit failed.");
  }
  if (EVP_DigestSignUpdate(md_ctx.get(), to_sign.data(), to_sign.size()) != 1) {
    return absl::InternalError("DigestUpdate failed.");
  }
  size_t sig_len = 0;
  if (EVP_DigestSignFinal(md_ctx.get(), nullptr, &sig_len) != 1) {
    return absl::InternalError("DigestFinal (get signature length) failed.");
  }

  std::unique_ptr<unsigned char[]> sig(new unsigned char[sig_len]);
  if (EVP_DigestSignFinal(md_ctx.get(), sig.get(), &sig_len) != 1) {
    return absl::InternalError("DigestFinal (signature compute) failed.");
  }

  std::string signature;
  absl::WebSafeBase64Escape(
      std::string_view(reinterpret_cast<char*>(sig.get()), sig_len),
      &signature);

  return std::move(signature);
}

/// Encodes a claim for a JSON web token (JWT) to make an OAuth request.
std::string BuildJWTHeader(std::string_view key_id) {
  // 1. Create the assertion header.
  ::nlohmann::json assertion_header = {
      {"alg", kCryptoAlgorithm},
      {"typ", kJwtType},
      {"kid", std::string(key_id)},
  };

  std::string encoded_header;
  absl::WebSafeBase64Escape(assertion_header.dump(), &encoded_header);
  return encoded_header;
}

std::string BuildJWTClaimBody(std::string_view client_email,
                              std::string_view scope,  //
                              std::string_view audience, absl::Time now,
                              std::int64_t lifetime) {
  const std::int64_t request_timestamp_sec = absl::ToUnixSeconds(now);
  const std::int64_t expiration_timestamp_sec =
      request_timestamp_sec + lifetime;

  // 2. Create the assertion payload.
  ::nlohmann::json assertion_payload = {
      {"iss", std::string(client_email)}, {"scope", std::string(scope)},
      {"aud", std::string(audience)},     {"iat", request_timestamp_sec},
      {"exp", expiration_timestamp_sec},
  };

  std::string encoded_payload;
  absl::WebSafeBase64Escape(assertion_payload.dump(), &encoded_payload);
  return encoded_payload;
}

Result<std::string> BuildSignedJWTRequest(std::string_view private_key,
                                          std::string_view header,
                                          std::string_view body) {
  auto claim = tensorstore::StrCat(header, ".", body);
  auto result = SignWithRSA256(private_key, claim);
  if (!result) {
    return result.status();
  }
  return tensorstore::StrCat("grant_type=", kGrantType, "&assertion=", claim,
                             ".", *result);
}

constexpr static auto ErrorResponseBinder = jb::Object(
    jb::Member("error",
               jb::Projection(&ErrorResponse::error, jb::NonEmptyStringBinder)),
    jb::Member("error_description",
               jb::Projection(&ErrorResponse::error_description,
                              jb::NonEmptyStringBinder)),
    jb::Member("error_uri", jb::Projection(&ErrorResponse::error_uri,
                                           jb::NonEmptyStringBinder)),
    jb::Member("error_subtype", jb::Projection(&ErrorResponse::error_subtype,
                                               jb::NonEmptyStringBinder)),
    jb::DiscardExtraMembers);

Result<ErrorResponse> ParseErrorResponse(const ::nlohmann::json& error) {
  if (error.is_discarded()) {
    return absl::InvalidArgumentError("Invalid ErrorResponse");
  }
  return jb::FromJson<ErrorResponse>(error, ErrorResponseBinder);
}

constexpr static auto GoogleServiceAccountCredentialsBinder = jb::Object(
    jb::Member("private_key",
               jb::Projection(&GoogleServiceAccountCredentials::private_key,
                              jb::NonEmptyStringBinder)),
    jb::Member("private_key_id",
               jb::Projection(&GoogleServiceAccountCredentials::private_key_id,
                              jb::NonEmptyStringBinder)),
    jb::Member("client_email",
               jb::Projection(&GoogleServiceAccountCredentials::client_email,
                              jb::NonEmptyStringBinder)),
    jb::Member("token_uri",
               jb::Projection(&GoogleServiceAccountCredentials::token_uri,
                              jb::DefaultInitializedValue())),
    jb::DiscardExtraMembers);

Result<GoogleServiceAccountCredentials>
ParseGoogleServiceAccountCredentialsImpl(const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(
        "Invalid GoogleServiceAccountCredentials token");
  }

  // Google ServiceAccountCredentials files contain numerous fields that we
  // don't care to parse, such as { "type", "project_id", "client_id",
  // "auth_uri", "auth_provider_x509_cert_url", "client_x509_cert_url"}.
  auto creds_token = jb::FromJson<GoogleServiceAccountCredentials>(
      credentials, GoogleServiceAccountCredentialsBinder);
  if (!creds_token.ok()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid GoogleServiceAccountCredentials: ", creds_token.status()));
  }
  return creds_token;
}

Result<GoogleServiceAccountCredentials> ParseGoogleServiceAccountCredentials(
    std::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid GoogleServiceAccountCredentials: ", source));
  }
  return ParseGoogleServiceAccountCredentialsImpl(credentials);
}

constexpr static auto RefreshTokenBinder = jb::Object(
    jb::Member("client_id", jb::Projection(&RefreshToken::client_id,
                                           jb::NonEmptyStringBinder)),
    jb::Member("client_secret", jb::Projection(&RefreshToken::client_secret,
                                               jb::NonEmptyStringBinder)),
    jb::Member("refresh_token", jb::Projection(&RefreshToken::refresh_token,
                                               jb::NonEmptyStringBinder)),
    jb::DiscardExtraMembers);

Result<RefreshToken> ParseRefreshTokenImpl(
    const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::UnauthenticatedError("Invalid RefreshToken token");
  }
  auto refresh_token =
      jb::FromJson<RefreshToken>(credentials, RefreshTokenBinder);
  if (!refresh_token.ok()) {
    return absl::UnauthenticatedError(
        tensorstore::StrCat("Invalid RefreshToken: ", credentials.dump()));
  }
  return refresh_token;
}

Result<RefreshToken> ParseRefreshToken(std::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::UnauthenticatedError(
        tensorstore::StrCat("Invalid RefreshToken: ", source));
  }
  return ParseRefreshTokenImpl(credentials);
}

constexpr static auto OAuthResponseBinder = jb::Object(
    jb::Member("token_type", jb::Projection(&OAuthResponse::token_type,
                                            jb::NonEmptyStringBinder)),
    jb::Member("access_token", jb::Projection(&OAuthResponse::access_token,
                                              jb::NonEmptyStringBinder)),
    jb::Member("expires_in", jb::Projection(&OAuthResponse::expires_in,
                                            jb::LooseInteger<int64_t>(1))),
    jb::DiscardExtraMembers);

Result<OAuthResponse> ParseOAuthResponseImpl(
    const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::UnauthenticatedError("Invalid OAuthResponse token");
  }
  auto response_token =
      jb::FromJson<OAuthResponse>(credentials, OAuthResponseBinder);
  if (!response_token.ok()) {
    return absl::UnauthenticatedError(
        tensorstore::StrCat("Invalid OAuthResponse: ", credentials.dump()));
  }
  return response_token;
}

Result<OAuthResponse> ParseOAuthResponse(std::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::UnauthenticatedError(
        tensorstore::StrCat("Invalid OAuthResponse: ", source));
  }
  return ParseOAuthResponseImpl(credentials);
}

}  // namespace internal_oauth2
}  // namespace tensorstore
