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

#ifndef TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
#define TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_

/// \file
/// S3 Request Builder

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorstore/internal/ascii_utils.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/internal/digest/sha256.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/util/result.h"

#include "absl/strings/str_split.h"
#include "absl/time/time.h"

using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal::IntToHexDigit;
using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::SHA256Digester;

namespace tensorstore {
namespace internal_storage_s3 {

std::string UriEncode(std::string_view src);
std::string UriObjectKeyEncode(std::string_view src);

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



class S3RequestBuilder {
 public:
  explicit S3RequestBuilder(std::string_view method, std::string endpoint_url) :
    builder_(method, endpoint_url, UriEncode) {};

  /// Adds a prefix to the user-agent string.
  S3RequestBuilder & AddUserAgentPrefix(std::string_view prefix) {
    builder_.AddUserAgentPrefix(prefix);
    return *this;
  }

  /// Adds request headers.
  S3RequestBuilder & AddHeader(std::string header) {
    builder_.AddHeader(header);
    return *this;
  }

  /// Adds a parameter for a request.
  S3RequestBuilder& AddQueryParameter(std::string_view key, std::string_view value) {
    builder_.AddQueryParameter(key, value);
    return *this;
  }

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  S3RequestBuilder& EnableAcceptEncoding() {
    builder_.EnableAcceptEncoding();
    return *this;
  }

  /// Adds a `range` header to the http request if the byte_range
  /// is specified.
  S3RequestBuilder& AddRangeHeader(OptionalByteRangeRequest byte_range, bool & result) {
    builder_.AddRangeHeader(byte_range, result);
    return *this;
  };
  /// Adds a `cache-control` header specifying `max-age` or `no-cache`.
  S3RequestBuilder& AddCacheControlMaxAgeHeader(absl::Duration max_age, bool & result) {
    builder_.AddCacheControlMaxAgeHeader(max_age, result);
    return *this;
  };
  /// Adds a `cache-control` header consistent with `staleness_bound`.
  S3RequestBuilder& AddStalenessBoundCacheControlHeader(absl::Time staleness_bound, bool & result) {
    builder_.AddStalenessBoundCacheControlHeader(staleness_bound, result);
    return *this;
  }


  HttpRequest BuildRequest(std::string_view aws_access_key, std::string_view aws_secret_access_key,
                           std::string_view aws_region, std::string_view payload_hash,
                           const absl::Time & time);

  /// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
  static std::string SigningString(
    std::string_view canonical_request,
    std::string_view aws_region,
    const absl::Time & time);

  static std::string Signature(
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    std::string_view signing_string,
    const absl::Time & time);

  static std::string CanonicalRequest(
    std::string_view url,
    std::string_view method,
    std::string_view payload_hash,
    const std::vector<std::string> & headers,
    const std::vector<std::pair<std::string, std::string>> & queries);

  static std::string AuthorizationHeader(
    std::string_view aws_access_key,
    std::string_view aws_region,
    std::string_view signature,
    const std::vector<std::string> & headers,
    const absl::Time & time);
 private:
  HttpRequestBuilder builder_;
  char const* query_parameter_separator_;
};

} // namespace internal_storage_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
