// Copyright 2023 The TensorStore Authors
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

#ifdef _WIN32
// openssl/boringssl may have conflicting macros with Windows.
#define WIN32_LEAN_AND_MEAN
#endif

#include "tensorstore/kvstore/s3/s3_request_builder.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include <openssl/evp.h>  // IWYU pragma: keep
#include <openssl/hmac.h>
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/kvstore/s3/s3_uri_utils.h"

using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::SHA256Digester;
using ::tensorstore::internal_http::HttpRequest;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag s3_logging("s3");

/// Size of HMAC (size of SHA256 digest).
constexpr static size_t kHmacSize = 32;

void ComputeHmac(std::string_view key, std::string_view message,
                 unsigned char (&hmac)[kHmacSize]) {
  unsigned int md_len = kHmacSize;
  // Computing HMAC should never fail.
  ABSL_CHECK(
      HMAC(EVP_sha256(), reinterpret_cast<const unsigned char*>(key.data()),
           key.size(), reinterpret_cast<const unsigned char*>(message.data()),
           message.size(), hmac, &md_len) &&
      md_len == kHmacSize);
}

void ComputeHmac(unsigned char (&key)[kHmacSize], std::string_view message,
                 unsigned char (&hmac)[kHmacSize]) {
  unsigned int md_len = kHmacSize;
  // Computing HMAC should never fail.
  ABSL_CHECK(HMAC(EVP_sha256(), key, kHmacSize,
                  reinterpret_cast<const unsigned char*>(message.data()),
                  message.size(), hmac, &md_len) &&
             md_len == kHmacSize);
}

std::pair<std::string_view, std::string_view> SplitAuthorityAndPath(
    std::string_view authority_and_path) {
  size_t end_of_authority = authority_and_path.find('/');
  if (end_of_authority == std::string_view::npos) {
    return {authority_and_path, {}};
  }

  return {authority_and_path.substr(0, end_of_authority),
          authority_and_path.substr(end_of_authority)};
}

/// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
std::string CanonicalRequest(
    std::string_view method, std::string_view path, std::string_view query,
    std::string_view payload_hash,
    const std::vector<std::pair<std::string, std::string_view>>& headers) {
  std::string canonical =
      absl::StrCat(method, "\n", S3UriObjectKeyEncode(path), "\n", query, "\n");

  // Canonical Headers
  std::vector<std::string_view> signed_headers;
  signed_headers.reserve(headers.size());
  for (auto& pair : headers) {
    absl::StrAppend(&canonical, pair.first, ":", pair.second, "\n");
    signed_headers.push_back(pair.first);
  }
  // Signed Headers
  absl::StrAppend(&canonical, "\n", absl::StrJoin(signed_headers, ";"), "\n",
                  payload_hash);
  return canonical;
}

std::string SigningString(std::string_view canonical_request,
                          std::string_view aws_region, const absl::Time& time) {
  absl::TimeZone utc = absl::UTCTimeZone();
  SHA256Digester sha256;
  sha256.Write(canonical_request);
  const auto digest = sha256.Digest();
  auto digest_sv = std::string_view(reinterpret_cast<const char*>(&digest[0]),
                                    digest.size());

  return absl::StrFormat(
      "AWS4-HMAC-SHA256\n"
      "%s\n"
      "%s/%s/s3/aws4_request\n"
      "%s",
      absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc),
      absl::FormatTime("%Y%m%d", time, utc), aws_region,
      absl::BytesToHexString(digest_sv));
}

std::string Signature(std::string_view aws_secret_access_key,
                      std::string_view aws_region,
                      std::string_view signing_string, const absl::Time& time) {
  absl::TimeZone utc = absl::UTCTimeZone();
  unsigned char date_key[kHmacSize];
  unsigned char date_region_key[kHmacSize];
  unsigned char date_region_service_key[kHmacSize];
  unsigned char signing_key[kHmacSize];
  unsigned char final_key[kHmacSize];

  ComputeHmac(absl::StrCat("AWS4", aws_secret_access_key),
              absl::FormatTime("%Y%m%d", time, utc), date_key);
  ComputeHmac(date_key, aws_region, date_region_key);
  ComputeHmac(date_region_key, "s3", date_region_service_key);
  ComputeHmac(date_region_service_key, "aws4_request", signing_key);
  ComputeHmac(signing_key, signing_string, final_key);

  auto key_view =
      std::string_view(reinterpret_cast<const char*>(final_key), kHmacSize);
  return absl::BytesToHexString(key_view);
}

std::string AuthorizationHeader(
    std::string_view aws_access_key, std::string_view aws_region,
    std::string_view signature,
    const std::vector<std::pair<std::string, std::string_view>>& headers,
    const absl::Time& time) {
  return absl::StrFormat(
      "Authorization: AWS4-HMAC-SHA256 Credential=%s"
      "/%s/%s/s3/aws4_request,"
      "SignedHeaders=%s,Signature=%s",
      aws_access_key, absl::FormatTime("%Y%m%d", time, absl::UTCTimeZone()),
      aws_region,
      absl::StrJoin(headers, ";",
                    [](std::string* out, auto pair) {
                      absl::StrAppend(out, pair.first);
                    }),
      signature);
}

static constexpr char kAmzContentSha256Header[] = "x-amz-content-sha256: ";
static constexpr char kAmzSecurityTokenHeader[] = "x-amz-security-token: ";
/// https://docs.aws.amazon.com/AmazonS3/latest/userguide/ObjectsinRequesterPaysBuckets.html
/// For DELETE, GET, HEAD, POST, and PUT requests, include x-amz-request-payer :
/// requester in the header
static constexpr char kAmzRequesterPayerHeader[] =
    "x-amz-requester-payer: requester";

}  // namespace

S3RequestBuilder& S3RequestBuilder::MaybeAddRequesterPayer(
    bool requester_payer) {
  if (requester_payer) {
    builder_.AddHeader(kAmzRequesterPayerHeader);
  }
  return *this;
}

HttpRequest S3RequestBuilder::BuildRequest(std::string_view host_header,
                                           const AwsCredentials& credentials,
                                           std::string_view aws_region,
                                           std::string_view payload_sha256_hash,
                                           const absl::Time& time) {
  builder_.AddHostHeader(host_header);
  builder_.AddHeader(
      absl::StrCat(kAmzContentSha256Header, payload_sha256_hash));
  builder_.AddHeader(absl::FormatTime("x-amz-date: %Y%m%dT%H%M%SZ", time,
                                      absl::UTCTimeZone()));

  // Add deferred query parameters in sorted order for AWS4 signature
  // requirements
  std::stable_sort(std::begin(query_params_), std::end(query_params_));
  for (const auto& [k, v] : query_params_) {
    builder_.AddQueryParameter(k, v);
  }

  // If anonymous, it's unnecessary to construct the Authorization header
  if (credentials.IsAnonymous()) {
    return builder_.BuildRequest();
  }

  // Add AWS Session Token, if available
  // https://docs.aws.amazon.com/AmazonS3/latest/userguide/RESTAuthentication.html#UsingTemporarySecurityCredentials
  if (!credentials.session_token.empty()) {
    builder_.AddHeader(
        absl::StrCat(kAmzSecurityTokenHeader, credentials.session_token));
  }

  auto request = builder_.BuildRequest();

  // Create sorted AWS4 signing headers
  std::vector<std::pair<std::string, std::string_view>> signed_headers;
  signed_headers.reserve(request.headers.size());
  for (const auto& header_str : request.headers) {
    std::string_view header = header_str;
    auto pos = header.find(':');
    assert(pos != std::string::npos);
    auto key = absl::AsciiStrToLower(
        absl::StripAsciiWhitespace(header.substr(0, pos)));
    auto value = absl::StripAsciiWhitespace(header.substr(pos + 1));
    signed_headers.push_back({std::move(key), std::move(value)});
  }
  std::stable_sort(std::begin(signed_headers), std::end(signed_headers));

  auto parsed_uri = ParseGenericUri(request.url);
  auto authority_and_path =
      SplitAuthorityAndPath(parsed_uri.authority_and_path);
  assert(!authority_and_path.second.empty());

  canonical_request_ =
      CanonicalRequest(request.method, authority_and_path.second,
                       parsed_uri.query, payload_sha256_hash, signed_headers);
  signing_string_ = SigningString(canonical_request_, aws_region, time);
  signature_ =
      Signature(credentials.secret_key, aws_region, signing_string_, time);
  auto auth_header = AuthorizationHeader(credentials.access_key, aws_region,
                                         signature_, signed_headers, time);

  ABSL_LOG_IF(INFO, s3_logging.Level(1))  //
      << "Canonical Request\n"
      << canonical_request_  //
      << "\n\nSigning String\n"
      << signing_string_  //
      << "\n\nAuthorization Header\n"
      << auth_header;

  request.headers.emplace_back(std::move(auth_header));
  return request;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
