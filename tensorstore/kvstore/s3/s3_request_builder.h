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
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/kvstore/s3/s3_uri_utils.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// @brief Builds an HTTP Request for submission to an S3 Endpoint
///
/// S3RequestBuilder encapsulates a HttpRequestBuilder to
/// build HTTP Requests specific to an S3 endpoint, in order
/// to interact with the S3 REST API
/// https://docs.aws.amazon.com/AmazonS3/latest/API/Type_API_Reference.html
///
/// It adds the following functionality to HttpRequestBuilder:
///
///   1. The *host*, *x-amz-content-sha256* and *x-amz-date* headers are added
///      to the request headers when `BuildRequest` is called.
///   2. If provided with S3 credentials, an Authorization header is added to
///      the request headers.
///      Additionally, an *x-amz-security-token* header is added if an STS
///      session token is provided in the credentials. The calculation of this
///      header is described here.
///      https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
///   3. If the AWS credentials are empty, the Authorization header is omitted,
///      representing anonymous access.
///   4. The request url and query parameters are encoded using S3 specific URI
///      encoding logic described here:
///      https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html#create-signature-presign-entire-payload
///
class S3RequestBuilder {
 public:
  /// Constructs an S3RequestBuilder with the HTTP Method (e.g. GET, PUT,
  /// DELETE, HEAD) and the S3 endpoint
  S3RequestBuilder(std::string_view method, std::string endpoint_url)
      : builder_(method, std::move(endpoint_url), S3UriEncode) {}

  /// Adds request headers.
  S3RequestBuilder& AddHeader(std::string_view header) {
    builder_.AddHeader(header);
    return *this;
  }

  /// Adds a query parameter for a request.
  S3RequestBuilder& AddQueryParameter(std::string key, std::string value) {
    query_params_.push_back({std::move(key), std::move(value)});
    return *this;
  }

  /// Enables sending Accept-Encoding header and transparently decoding the
  /// response.
  S3RequestBuilder& EnableAcceptEncoding() {
    builder_.EnableAcceptEncoding();
    return *this;
  }

  /// Adds a requester payer header to the requester if `requester_payer` is
  /// true
  S3RequestBuilder& MaybeAddRequesterPayer(bool requester_payer = false);

  /// Adds a `range` header to the http request if the byte_range
  /// is specified.
  S3RequestBuilder& MaybeAddRangeHeader(OptionalByteRangeRequest byte_range) {
    builder_.MaybeAddRangeHeader(byte_range);
    return *this;
  }

  /// Adds a `cache-control` header specifying `max-age` or `no-cache`.
  S3RequestBuilder& MaybeAddCacheControlMaxAgeHeader(absl::Duration max_age) {
    builder_.MaybeAddCacheControlMaxAgeHeader(max_age);
    return *this;
  }

  /// Adds a `cache-control` header consistent with `staleness_bound`.
  S3RequestBuilder& MaybeAddStalenessBoundCacheControlHeader(
      absl::Time staleness_bound) {
    builder_.MaybeAddStalenessBoundCacheControlHeader(staleness_bound);
    return *this;
  }

  /// Provides the Canonical Request artifact once BuildRequest has been called
  const std::string& GetCanonicalRequest() const { return canonical_request_; }

  /// Provides the Signing String artifact once BuildRequest has been called
  const std::string& GetSigningString() const { return signing_string_; }

  /// Provides the GetSignature artifact once BuildRequest has been called
  const std::string& GetSignature() const { return signature_; }

  /// Builds an HTTP Request given the information provided to the builder
  ///
  /// The `host_header` should be the header value described here
  /// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Host
  /// `credentials` contains the keys required to construct an Authorization
  /// header with an AWS4 signature. An empty access key on `credentials`
  /// implies anonymous access.
  internal_http::HttpRequest BuildRequest(std::string_view host_header,
                                          const AwsCredentials& credentials,
                                          std::string_view aws_region,
                                          std::string_view payload_sha256_hash,
                                          const absl::Time& time);

 private:
  std::string canonical_request_;
  std::string signing_string_;
  std::string signature_;
  std::vector<std::pair<std::string, std::string>> query_params_;
  internal_http::HttpRequestBuilder builder_;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_REQUEST_BUILDER_H_
