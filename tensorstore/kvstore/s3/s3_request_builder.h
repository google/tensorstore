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

#ifndef TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
#define TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_

/// \file
/// S3 Request Builder

#include <string>
#include <string_view>
#include <vector>

#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/s3/s3_uri_utils.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/util/result.h"

#include "absl/time/time.h"

using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http_s3::S3UriEncode;
using ::tensorstore::internal_http_s3::S3UriObjectKeyEncode;

namespace tensorstore {
namespace internal_storage_s3 {

class S3RequestBuilder {
 public:
  S3RequestBuilder(std::string_view method, std::string endpoint_url) :
    builder_(method, endpoint_url, S3UriEncode) {};


  /// Adds request headers.
  S3RequestBuilder & AddHeader(std::string_view header) {
    builder_.AddHeader(header);
    return *this;
  }

  /// Adds a parameter for a request.
  S3RequestBuilder& AddQueryParameter(std::string_view key, std::string_view value) {
    query_params_.push_back({std::string(key), std::string(value)});
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
  S3RequestBuilder& MaybeAddRangeHeader(OptionalByteRangeRequest byte_range) {
    builder_.MaybeAddRangeHeader(byte_range);
    return *this;
  };
  /// Adds a `cache-control` header specifying `max-age` or `no-cache`.
  S3RequestBuilder& MaybeAddCacheControlMaxAgeHeader(absl::Duration max_age) {
    builder_.MaybeAddCacheControlMaxAgeHeader(max_age);
    return *this;
  };
  /// Adds a `cache-control` header consistent with `staleness_bound`.
  S3RequestBuilder& MaybeAddStalenessBoundCacheControlHeader(absl::Time staleness_bound) {
    builder_.MaybeAddStalenessBoundCacheControlHeader(staleness_bound);
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
    const std::vector<std::pair<std::string, std::string>> & headers);

  static std::string AuthorizationHeader(
    std::string_view aws_access_key,
    std::string_view aws_region,
    std::string_view signature,
    const std::vector<std::pair<std::string, std::string>> & headers,
    const absl::Time & time);
 private:
  std::vector<std::pair<std::string, std::string>> query_params_;
  HttpRequestBuilder builder_;
};

} // namespace internal_storage_s3
} // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
