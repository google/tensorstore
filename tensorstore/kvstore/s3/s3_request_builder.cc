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

#include <algorithm>
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/internal/ascii_utils.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal::AsciiSet;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::ParsedGenericUri;
using ::tensorstore::internal_storage_s3::IsValidBucketName;

namespace tensorstore {
namespace internal_storage_s3 {


namespace {

// See description of function UriEncode at this URL
// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
constexpr AsciiSet kUriUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~"};

// NOTE: Only adds "/" to kUriUnreservedChars
constexpr AsciiSet kUriObjectKeyReservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "/-._~"};
}


std::string UriEncode(std::string_view src) {
    std::string dest;
    PercentEncodeReserved(src, dest, kUriUnreservedChars);
    return dest;
}

std::string UriObjectKeyEncode(std::string_view src) {
    std::string dest;
    PercentEncodeReserved(src, dest, kUriObjectKeyReservedChars);
    return dest;
}

S3RequestBuilder::S3RequestBuilder(std::string_view method,
                                   std::string endpoint_url)
    : query_parameter_separator_("?") {
  request_.method_ = method;
  request_.url_ = std::move(endpoint_url);
}

HttpRequest S3RequestBuilder::BuildRequest(
    std::string_view aws_access_key,
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    std::string_view payload_hash,
    const absl::Time & time) {

  auto url = std::string_view(request_.url());
  auto query_pos = url.find('?');
  std::vector<std::pair<std::string, std::string>> queries;

  if(query_pos != std::string::npos && query_pos + 1 != std::string::npos) {
    auto query = url.substr(query_pos + 1);
    std::vector<std::string_view> query_bits = absl::StrSplit(query, '&');

    for(auto query_bit: query_bits) {
      std::vector<std::string_view> key_values = absl::StrSplit(query_bit, '=');
      assert(key_values.size() == 1 || key_values.size() == 2);

      if(key_values.size() == 1) {
        queries.push_back({std::string(key_values[0]), ""});
      } else if(key_values.size() == 2) {
        queries.push_back({std::string(key_values[0]), std::string(key_values[1])});
      }
    }
  }

  auto headers = request_.headers();
  std::sort(std::begin(headers), std::end(headers));
  std::sort(std::begin(queries), std::end(queries));
  auto canonical_request = CanonicalRequest(request_.url(), request_.method(),
                                            payload_hash, headers,
                                            queries);
  auto signing_string = SigningString(canonical_request, aws_region, time);
  auto signature = Signature(aws_secret_access_key, aws_region, signing_string, time);
  auto auth_header = AuthorizationHeader(aws_access_key, aws_region, signature, headers, time);
  request_.headers_.emplace_back(std::move(auth_header));
  return std::move(request_);
}

S3RequestBuilder& S3RequestBuilder::AddUserAgentPrefix(std::string_view prefix) {
  request_.user_agent_ = tensorstore::StrCat(prefix, request_.user_agent_);
  return *this;
}

S3RequestBuilder& S3RequestBuilder::AddHeader(std::string header) {
  request_.headers_.emplace_back(std::move(header));
  return *this;
}

S3RequestBuilder& S3RequestBuilder::AddQueryParameter(
    std::string_view key, std::string_view value) {
  auto enc_value = UriEncode(value);
  auto enc_key = UriEncode(key);
  auto parameter = tensorstore::StrCat(
      query_parameter_separator_,
      enc_key, "=", enc_value);
  query_parameter_separator_ = "&";
  request_.url_.append(parameter);
  return *this;
}

S3RequestBuilder& S3RequestBuilder::EnableAcceptEncoding() {
  request_.accept_encoding_ = true;
  return *this;
}

bool AddRangeHeader(S3RequestBuilder& request_builder,
                    OptionalByteRangeRequest byte_range) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Range
  if (byte_range.exclusive_max) {
    assert(byte_range.SatisfiesInvariants());
    request_builder.AddHeader(absl::StrFormat("Range: bytes=%d-%d",
                                              byte_range.inclusive_min,
                                              *byte_range.exclusive_max - 1));
    return true;
  }
  if (byte_range.inclusive_min > 0) {
    request_builder.AddHeader(
        absl::StrFormat("Range: bytes=%d-", byte_range.inclusive_min));
    return true;
  }
  return false;
}

bool AddCacheControlMaxAgeHeader(S3RequestBuilder& request_builder,
                                 absl::Duration max_age) {
  // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control
  if (max_age >= absl::InfiniteDuration()) {
    return false;
  }

  // Since max-age is specified as an integer number of seconds, always
  // round down to ensure our requirement is met.
  auto max_age_seconds = absl::ToInt64Seconds(max_age);
  if (max_age_seconds > 0) {
    request_builder.AddHeader(
        absl::StrFormat("cache-control: max-age=%d", max_age_seconds));
  } else {
    request_builder.AddHeader("cache-control: no-cache");
  }
  return true;
}

bool AddStalenessBoundCacheControlHeader(S3RequestBuilder& request_builder,
                                         absl::Time staleness_bound) {
  // `absl::InfiniteFuture()`  indicates that the result must be current.
  // `absl::InfinitePast()`  disables the cache-control header.
  if (staleness_bound == absl::InfinitePast()) {
    return false;
  }
  absl::Time now;
  absl::Duration duration = absl::ZeroDuration();
  if (staleness_bound != absl::InfiniteFuture() &&
      (now = absl::Now()) > staleness_bound) {
    // the max age is in the past.
    duration = now - staleness_bound;
  }
  return AddCacheControlMaxAgeHeader(request_builder, duration);
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
    reinterpret_cast<const char *>(digest.begin()),
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

  ComputeHmac(absl::StrFormat("AWS4%s",aws_secret_access_key),
              absl::FormatTime("%Y%m%d", time, utc), date_key);
  ComputeHmac(date_key, aws_region, date_region_key);
  ComputeHmac(date_region_key, "s3", date_region_service_key);
  ComputeHmac(date_region_service_key, "aws4_request", signing_key);
  ComputeHmac(signing_key, signing_string, final_key);

  std::string result(2 * kHmacSize, '0');

  for(int i=0; i < kHmacSize; ++i) {
      result[2*i + 0] = IntToHexDigit(final_key[i] / 16, false);
      result[2*i + 1] = IntToHexDigit(final_key[i] % 16, false);
  }

  return result;
}

std::string S3RequestBuilder::CanonicalRequest(
  std::string_view url,
  std::string_view method,
  std::string_view payload_hash,
  const std::vector<std::string> & headers,
  const std::vector<std::pair<std::string, std::string>> & queries)
{
  auto uri = ParseGenericUri(url);
  std::size_t end_of_bucket = uri.authority_and_path.find('/');
  assert(end_of_bucket != std::string_view::npos);
  auto path = uri.authority_and_path.substr(end_of_bucket);
  assert(path.size() > 0);

  absl::Cord cord;
  cord.Append(method);
  cord.Append("\n");
  cord.Append(UriObjectKeyEncode(path));
  cord.Append("\n");

  // Query string
  for(auto [it, first] = std::tuple{queries.begin(), true}; it != queries.end(); ++it, first=false) {
    if(!first) cord.Append("&");
    cord.Append(UriEncode(it->first));
    cord.Append("=");
    cord.Append(UriEncode(it->second));
  }

  cord.Append("\n");

  // Headers
  for(auto header: headers) {
    auto delim_pos = header.find(':');
    assert(delim_pos != std::string::npos && delim_pos + 1 != std::string::npos);
    cord.Append(header.substr(0, delim_pos));
    cord.Append(":");
    cord.Append(absl::StripAsciiWhitespace(header.substr(delim_pos + 1)));
    cord.Append("\n");
  }

  cord.Append("\n");

  // Signed headers
  for(auto [it, first] = std::tuple{headers.begin(), true}; it != headers.end(); ++it, first=false) {
    if(!first) cord.Append(";");
    auto header = std::string_view(*it);
    auto delim_pos = header.find(':');
    assert(delim_pos != std::string::npos);
    cord.Append(header.substr(0, delim_pos));
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
  const std::vector<std::string> & headers,
  const absl::Time & time) {

  std::vector<std::string_view> header_names;
  for(auto & header: headers) {
    std::size_t delim_pos = header.find(':');
    assert(delim_pos != std::string::npos);
    header_names.push_back(std::string_view(header).substr(0, delim_pos));
  }

  return absl::StrFormat(
    "Authorization: AWS4-HMAC-SHA256 Credential=%s"
    "/%s/%s/s3/aws4_request,"
    "SignedHeaders=%s,Signature=%s",
      aws_access_key,
      absl::FormatTime("%Y%m%d", time, absl::UTCTimeZone()),
      aws_region,
      absl::StrJoin(header_names, ";"),
      signature
  );
}


} // namespace internal_storage_s3
} // namespace tensorstore
