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

#include <algorithm>
#include <string>
#include <string_view>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "absl/status/status.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal::AsciiSet;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::ParsedGenericUri;
using ::tensorstore::internal::SHA256Digester;
using namespace ::tensorstore::internal_http;
using ::tensorstore::internal_storage_s3::IsValidBucketName;

#ifndef TENSORSTORE_INTERNAL_S3_LOG_AWS4
#define TENSORSTORE_INTERNAL_S3_LOG_AWS4 1
#endif

namespace tensorstore {
namespace internal_storage_s3 {

namespace {

/// Size of HMAC (size of SHA256 digest).
constexpr static size_t kHmacSize = 32;

void ComputeHmac(std::string_view key, std::string_view message, unsigned char (&hmac)[kHmacSize]) {
    unsigned int md_len = kHmacSize;
    // Computing HMAC should never fail.
    ABSL_CHECK(HMAC(EVP_sha256(),
                    reinterpret_cast<const unsigned char*>(key.data()),
                    key.size(),
                    reinterpret_cast<const unsigned char*>(message.data()),
                    message.size(), hmac, &md_len) &&
               md_len == kHmacSize);
}

void ComputeHmac(unsigned char (&key)[kHmacSize], std::string_view message, unsigned char (&hmac)[kHmacSize]){
    unsigned int md_len = kHmacSize;
    // Computing HMAC should never fail.
    ABSL_CHECK(HMAC(EVP_sha256(), key, kHmacSize,
                    reinterpret_cast<const unsigned char*>(message.data()),
                    message.size(), hmac, &md_len) &&
               md_len == kHmacSize);
}

} // namespace


HttpRequest S3RequestBuilder::BuildRequest(
    std::string_view aws_access_key,
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    std::string_view payload_hash,
    const absl::Time & time) {

  // Sort and add query parameters to the builder
  std::stable_sort(std::begin(query_params_), std::end(query_params_));
  for (const auto& [k, v] : query_params_) {
    builder_.AddQueryParameter(k, v);
  }

  auto request = builder_.BuildRequest();

  // Normalise headers
  std::vector<std::pair<std::string, std::string>> signed_headers;

  for(const auto & header_str: request.headers) {
    auto header = std::string_view(header_str);
    auto pos = header.find(':');
    assert(pos != std::string::npos);
    auto key = absl::AsciiStrToLower(absl::StripAsciiWhitespace(header.substr(0, pos)));
    auto value = absl::StripAsciiWhitespace(header.substr(pos + 1));
    signed_headers.push_back({std::string(key), std::string(value)});
  }

  std::stable_sort(std::begin(signed_headers), std::end(signed_headers));

  auto canonical_request = CanonicalRequest(request.url, request.method,
                                            payload_hash, signed_headers);
  auto signing_string = SigningString(canonical_request, aws_region, time);
  auto signature = Signature(aws_secret_access_key, aws_region, signing_string, time);
  auto auth_header = AuthorizationHeader(aws_access_key, aws_region, signature,
                                         signed_headers, time);

  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_AWS4)
      << "Canonical Request\n" << canonical_request;
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_AWS4)
      << "Signing String\n" << signing_string;
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_S3_LOG_AWS4)
      << "Authorization Header\n" << auth_header;

  request.headers.emplace_back(std::move(auth_header));
  return request;
}

/// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
std::string S3RequestBuilder::SigningString(
  std::string_view canonical_request,
  std::string_view aws_region,
  const absl::Time & time)
{
  absl::TimeZone utc = absl::UTCTimeZone();
  SHA256Digester sha256;
  sha256.Write(canonical_request);
  const auto digest = sha256.Digest();
  auto digest_sv = std::string_view(
    reinterpret_cast<const char *>(&digest[0]),
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

std::string S3RequestBuilder::Signature(
  std::string_view aws_secret_access_key,
  std::string_view aws_region,
  std::string_view signing_string,
  const absl::Time & time)
{
  absl::TimeZone utc = absl::UTCTimeZone();
  unsigned char date_key[kHmacSize];
  unsigned char date_region_key[kHmacSize];
  unsigned char date_region_service_key[kHmacSize];
  unsigned char signing_key[kHmacSize];
  unsigned char final_key[kHmacSize];

  ComputeHmac(absl::StrCat("AWS4",aws_secret_access_key),
              absl::FormatTime("%Y%m%d", time, utc), date_key);
  ComputeHmac(date_key, aws_region, date_region_key);
  ComputeHmac(date_region_key, "s3", date_region_service_key);
  ComputeHmac(date_region_service_key, "aws4_request", signing_key);
  ComputeHmac(signing_key, signing_string, final_key);

  auto key_view = std::string_view(reinterpret_cast<const char *>(final_key), kHmacSize);
  return absl::BytesToHexString(key_view);
}

std::string S3RequestBuilder::CanonicalRequest(
  std::string_view url,
  std::string_view method,
  std::string_view payload_hash,
  const std::vector<std::pair<std::string, std::string>> & headers)
{
  auto uri = ParseGenericUri(url);
  std::size_t end_of_bucket = uri.authority_and_path.find('/');
  assert(end_of_bucket != std::string_view::npos);
  auto path = uri.authority_and_path.substr(end_of_bucket);
  assert(path.size() > 0);

  absl::Cord cord;
  cord.Append(method);
  cord.Append("\n");
  cord.Append(S3UriObjectKeyEncode(path));
  cord.Append("\n");

  cord.Append(uri.query);
  cord.Append("\n");

  for(auto & pair: headers) {
    cord.Append(pair.first);
    cord.Append(":");
    cord.Append(pair.second);
    cord.Append("\n");
  }
  cord.Append("\n");

  // Signed headers
  for(auto [it, first] = std::tuple{headers.begin(), true}; it != headers.end(); ++it, first=false) {
    if(!first) cord.Append(";");
    cord.Append(it->first);
  }

  cord.Append("\n");
  cord.Append(payload_hash);

  std::string result;
  absl::CopyCordToString(cord, &result);
  return result;
}

std::string S3RequestBuilder::AuthorizationHeader(
    std::string_view aws_access_key,
    std::string_view aws_region,
    std::string_view signature,
    const std::vector<std::pair<std::string, std::string>> & headers,
    const absl::Time & time) {
  return absl::StrFormat(
    "Authorization: AWS4-HMAC-SHA256 Credential=%s"
    "/%s/%s/s3/aws4_request,"
    "SignedHeaders=%s,Signature=%s",
      aws_access_key,
      absl::FormatTime("%Y%m%d", time, absl::UTCTimeZone()),
      aws_region,
      absl::StrJoin(headers, ";",
                    [](std::string * out, auto pair) {
                      absl::StrAppend(out, pair.first);
                    }),
      signature
  );
}


} // namespace internal_storage_s3
} // namespace tensorstore
