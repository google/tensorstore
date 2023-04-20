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

#include "absl/strings/str_split.h"
#include "absl/strings/cord.h"
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



std::string S3RequestBuilder::CanonicalRequestBuilder::BuildCanonicalRequest() {
    auto uri = ParseGenericUri(url_);
    std::size_t end_of_bucket = uri.authority_and_path.find('/');
    assert(end_of_bucket != std::string_view::npos);
    auto path = uri.authority_and_path.substr(end_of_bucket);
    assert(path.size() > 0);

    absl::Cord cord;
    cord.Append(method_);
    cord.Append("\n");
    cord.Append(UriObjectKeyEncode(path));
    cord.Append("\n");

    // Query string
    for(auto [it, first] = std::tuple{queries_.begin(), true}; it != queries_.end(); ++it, first=false) {
     if(!first) cord.Append("&");
     cord.Append(UriEncode(it->first));
     cord.Append("=");
     cord.Append(UriEncode(it->second));
    }

    cord.Append("\n");

    // Headers
    for(auto it = headers_.begin(); it != headers_.end(); ++it) {
      cord.Append(it->first);
      cord.Append(":");
      cord.Append(absl::StripAsciiWhitespace(it->second));
      cord.Append("\n");
    }

    cord.Append("\n");

    // Signed headers
    for(auto [it, first] = std::tuple{headers_.begin(), true}; it != headers_.end(); ++it, first=false) {
      if(!first) cord.Append(";");
      cord.Append(it->first);
    }

    cord.Append("\n");
    assert(payload_hash_.has_value());
    cord.Append(payload_hash_.value());

    std::string result;
    absl::CopyCordToString(cord, &result);
    return result;
}


S3RequestBuilder::S3RequestBuilder(std::string_view method,
                                   std::string endpoint_url)
    : query_parameter_separator_("?"),
      canonical_builder(method, endpoint_url) {
  request_.method_ = method;
  request_.url_ = std::move(endpoint_url);
}

HttpRequest S3RequestBuilder::BuildRequest() { return std::move(request_); }

S3RequestBuilder& S3RequestBuilder::AddUserAgentPrefix(std::string_view prefix) {
  request_.user_agent_ = tensorstore::StrCat(prefix, request_.user_agent_);
  return *this;
}

S3RequestBuilder& S3RequestBuilder::AddHeader(std::string header) {
  std::size_t split_pos = header.find(':');
  assert(split_pos != std::string::npos);

  auto key = header.substr(0, split_pos + 1);
  auto value = absl::StripAsciiWhitespace(std::string_view(header).substr(split_pos + 1));

  assert(!key.empty());
  assert(!value.empty());

  canonical_builder.AddHeader(key, value);
  request_.headers_.emplace_back(std::move(header));

  return *this;
}

S3RequestBuilder& S3RequestBuilder::AddQueryParameter(
    std::string_view key, std::string_view value) {
  std::string enc_value = UriEncode(value);
  std::string parameter = tensorstore::StrCat(
      query_parameter_separator_,
      UriEncode(key), "=", enc_value);
  canonical_builder.AddQueryParameter(UriEncode(absl::AsciiStrToLower(key)), enc_value);
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

} // namespace internal_storage_s3
} // namespace tensorstore
