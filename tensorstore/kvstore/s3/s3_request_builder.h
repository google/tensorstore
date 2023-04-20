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

#include <map>
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
using ::tensorstore::internal::ParsedGenericUri;
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
  class CanonicalRequestBuilder {
   public:
    explicit CanonicalRequestBuilder(std::string_view method, std::string endpoint_url) :
     method_(method), url_(endpoint_url) {}

    CanonicalRequestBuilder& AddQueryParameter(std::string_view key, std::string_view value) {
      queries_.insert({std::string(key), std::string(value)});
      return *this;
    }

    CanonicalRequestBuilder& AddHeader(std::string_view key, std::string_view value) {
      headers_.insert({
        std::string(absl::AsciiStrToLower(key)),
        std::string(absl::StripAsciiWhitespace(value))});
      return *this;
    }

    CanonicalRequestBuilder& AddPayloadHash(std::string_view payload_hash) {
      payload_hash_ = std::string(payload_hash);
      return *this;
    }

    std::string BuildCanonicalRequest();

   private:
    std::string method_;
    std::string url_;
    std::optional<std::string> payload_hash_;
    std::map<std::string, std::string> queries_;
    std::map<std::string, std::string> headers_;
  };

  class AuthorizationHeaderBuilder {
   public:
    explicit AuthorizationHeaderBuilder(
          std::string_view aws_access_key,
          std::string_view aws_region,
          const absl::Time & time,
          std::string_view signature)
      : aws_access_key_(aws_access_key),
        aws_region_(aws_region),
        time_(time),
        signature_(signature) {}

    AuthorizationHeaderBuilder& AddHeaderName(std::string_view header_name) {
      header_names_.insert(std::string(header_name));
      return *this;
    }

    std::string BuildAuthorizationHeader() {
      return absl::StrFormat(
        "AWS4-HMAC-SHA256 Credential=%s"
        "/%s/%s/s3/aws4_request,"
        "SignedHeaders=%s,Signature=%s",
          aws_access_key_,
          absl::FormatTime("%Y%m%d", time_, absl::UTCTimeZone()),
          aws_region_,
          absl::StrJoin(header_names_, ";"),
          signature_
      );
    }

   private:
    std::string aws_access_key_;
    std::string aws_region_;
    absl::Time time_;
    std::string signature_;
    std::set<std::string> header_names_;
  };

 /// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
  static std::string SigningString(
    std::string_view canonical_request,
    const absl::Time & time,
    std::string_view aws_region)
  {
    absl::TimeZone utc = absl::UTCTimeZone();
    SHA256Digester sha256;
    sha256.Write(canonical_request);

    return absl::StrFormat(
      "AWS4-HMAC-SHA256\n"
      "%s\n"
      "%s/%s/s3/aws4_request\n"
      "%s",
        absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc),
        absl::FormatTime("%Y%m%d", time, utc), aws_region,
        sha256.HexDigest(false));
  }

  static std::string Signature(
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    const absl::Time & time,
    std::string_view signing_string)
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

 public:
  explicit S3RequestBuilder(std::string_view method, std::string endpoint_url);

  /// Adds a prefix to the user-agent string.
  S3RequestBuilder & AddUserAgentPrefix(std::string_view prefix);

  /// Adds request headers.
  S3RequestBuilder & AddHeader(std::string header);

  /// Adds a parameter for a request.
  S3RequestBuilder& AddQueryParameter(std::string_view key, std::string_view value);

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  S3RequestBuilder& EnableAcceptEncoding();

  S3RequestBuilder& AddAwsAccessKey(std::string_view aws_access_key) {
    aws_access_key_ = aws_access_key;
    return *this;
  }

  S3RequestBuilder& AddAwsSecretKey(std::string_view aws_secret_key) {
    aws_secret_key_ = aws_secret_key;
    return *this;
  }

  S3RequestBuilder& AddAwsRegion(std::string_view aws_region) {
    aws_region_ = aws_region;
    return *this;
  }

  S3RequestBuilder& AddTimeStamp(std::optional<absl::Time> timestamp) {
    timestamp_ = timestamp;
    return *this;
  }

  HttpRequest BuildRequest();

 private:
  HttpRequest request_;
  char const* query_parameter_separator_;
  std::string aws_access_key_;
  std::string aws_secret_key_;
  std::string aws_region_;
  std::optional<absl::Time> timestamp_;
  CanonicalRequestBuilder canonical_builder;
};

} // namespace internal_storage_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
