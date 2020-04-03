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
#include <utility>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include <openssl/bio.h>     // IWYU pragma: keep
#include <openssl/digest.h>  // IWYU pragma: keep
#include <openssl/evp.h>     // IWYU pragma: keep
#include <openssl/pem.h>     // IWYU pragma: keep
#include <openssl/rsa.h>     // IWYU pragma: keep
#include "tensorstore/internal/json.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::internal::JsonHandleObjectMember;
using tensorstore::internal::JsonRequireObjectMember;
using tensorstore::internal::JsonRequireValueAs;

namespace tensorstore {
namespace {

// The crypto algorithm to be used with OAuth.
constexpr char kCryptoAlgorithm[] = "RS256";

// The token type for the OAuth request.
constexpr char kJwtType[] = "JWT";

// The grant type for the OAuth request. Already URL-encoded for convenience.
constexpr char kGrantType[] =
    "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer";

// Parsing helper for strings.
struct JsonStringOp {
  std::string* result;
  Status operator()(const ::nlohmann::json& j) {
    return JsonRequireValueAs(j, result,
                              [](const std::string& x) { return !x.empty(); });
  }
};

}  // namespace
namespace internal_oauth2 {

Result<std::string> SignWithRSA256(absl::string_view private_key,
                                   absl::string_view to_sign) {
  if (private_key.empty()) {
    return absl::InternalError("No private key provided.");
  }

  const auto md = EVP_sha256();
  ABSL_ASSERT(md != nullptr);

  auto md_ctx = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
      EVP_MD_CTX_create(), &EVP_MD_CTX_free);
  ABSL_ASSERT(md_ctx != nullptr);

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
  EVP_MD_CTX_cleanup(md_ctx.get());

  std::string signature;
  absl::WebSafeBase64Escape(
      absl::string_view(reinterpret_cast<char*>(sig.get()), sig_len),
      &signature);

  return std::move(signature);
}

/// Encodes a claim for a JSON web token (JWT) to make an OAuth request.
std::string BuildJWTHeader(absl::string_view key_id) {
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

std::string BuildJWTClaimBody(absl::string_view client_email,
                              absl::string_view scope,  //
                              absl::string_view audience, absl::Time now,
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

Result<std::string> BuildSignedJWTRequest(absl::string_view private_key,
                                          absl::string_view header,
                                          absl::string_view body) {
  auto claim = absl::StrCat(header, ".", body);
  auto result = SignWithRSA256(private_key, claim);
  if (!result) {
    return GetStatus(result);
  }
  return absl::StrCat("grant_type=", kGrantType, "&assertion=", claim, ".",
                      *result);
}

Result<GoogleServiceAccountCredentials>
ParseGoogleServiceAccountCredentialsImpl(const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(
        "Invalid GoogleServiceAccountCredentials token");
  }

  // Google ServiceAccountCredentials files contain numerous fields that we
  // don't care to parse, such as { "type", "project_id", "client_id",
  // "auth_uri", "auth_provider_x509_cert_url", "client_x509_cert_url"}.
  GoogleServiceAccountCredentials result;

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "private_key", JsonStringOp{&result.private_key}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "private_key_id", JsonStringOp{&result.private_key_id}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "client_email", JsonStringOp{&result.client_email}));

  TENSORSTORE_RETURN_IF_ERROR(JsonHandleObjectMember(  //
      credentials, "token_uri", JsonStringOp{&result.token_uri}));

  return std::move(result);
}

Result<GoogleServiceAccountCredentials> ParseGoogleServiceAccountCredentials(
    absl::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid GoogleServiceAccountCredentials: ", source));
  }
  return ParseGoogleServiceAccountCredentialsImpl(credentials);
}

Result<RefreshToken> ParseRefreshTokenImpl(
    const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError("Invalid RefreshToken token");
  }

  RefreshToken result;

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "client_id", JsonStringOp{&result.client_id}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "client_secret", JsonStringOp{&result.client_secret}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "refresh_token", JsonStringOp{&result.refresh_token}));

  return std::move(result);
}

Result<RefreshToken> ParseRefreshToken(absl::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid RefreshToken: ", source));
  }
  return ParseRefreshTokenImpl(credentials);
}

Result<OAuthResponse> ParseOAuthResponseImpl(
    const ::nlohmann::json& credentials) {
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError("Invalid OAuthResponse token");
  }

  OAuthResponse result;

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "token_type", JsonStringOp{&result.token_type}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "access_token", JsonStringOp{&result.access_token}));

  TENSORSTORE_RETURN_IF_ERROR(JsonRequireObjectMember(  //
      credentials, "expires_in", [&](const ::nlohmann::json& j) -> Status {
        return JsonRequireValueAs(j, &result.expires_in,
                                  [](std::int64_t x) { return x > 0; });
      }));

  return std::move(result);
}

Result<OAuthResponse> ParseOAuthResponse(absl::string_view source) {
  auto credentials = internal::ParseJson(source);
  if (credentials.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid OAuthResponse: ", source));
  }
  return ParseOAuthResponseImpl(credentials);
}

}  // namespace internal_oauth2
}  // namespace tensorstore
