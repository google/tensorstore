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

#include "tensorstore/kvstore/s3/s3_endpoint.h"

#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/s3/validate.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;

// NOTE: Review the AWS cli options for additional supported features:
// https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
//
// We may need to revisit the host/endpoint requirements

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

static constexpr char kAmzBucketRegionHeader[] = "x-amz-bucket-region";

// https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html
//
// When composing virtual-host style addressing for s3, the following pattern
// is constructed:
//
// <bucket>.<service>.<aws_region>.amazonaws.com
//
// NOTE: If the bucket name contains a '.', then "you can't use
// virtual-host-style addressing over HTTPS, unless you perform your own
// certificate validation".
//
// https://aws.amazon.com/blogs/aws/amazon-s3-path-deprecation-plan-the-rest-of-the-story/
struct S3VirtualHostFormatter {
  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    // It's unclear whether there is any actual advantage in preferring the
    // region-specific virtual hostname over than just bucket.s3.amazonaws.com,
    // which has already been resolved.
    return absl::StrFormat("https://%s.s3.%s.amazonaws.com", bucket,
                           aws_region);
  }
};

// Construct a path-style hostname and host header.
struct S3PathFormatter {
  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    return absl::StrFormat("https://s3.%s.amazonaws.com/%s", aws_region,
                           bucket);
  }
};

struct S3CustomFormatter {
  std::string endpoint;

  std::string GetEndpoint(std::string_view bucket,
                          std::string_view aws_region) const {
    return absl::StrFormat("%s/%s", endpoint, bucket);
  }
};

template <typename Formatter>
struct ResolveHost {
  std::string bucket;
  Formatter formatter;

  void operator()(Promise<S3EndpointRegion> promise,
                  ReadyFuture<HttpResponse> ready) {
    if (!promise.result_needed()) return;

    auto& headers = ready.value().headers;
    if (auto it = headers.find(kAmzBucketRegionHeader); it != headers.end()) {
      promise.SetResult(S3EndpointRegion{
          formatter.GetEndpoint(bucket, it->second),
          it->second,
      });
    } else {
      promise.SetResult(absl::FailedPreconditionError(
          tensorstore::StrCat("bucket ", bucket, " does not exist")));
    }
  }
};

}  // namespace

std::variant<absl::Status, S3EndpointRegion> ValidateEndpoint(
    std::string_view bucket, std::string aws_region, std::string_view endpoint,
    std::string host_header) {
  ABSL_CHECK(!bucket.empty());

  if (!host_header.empty() && endpoint.empty()) {
    return absl::InvalidArgumentError(
        "\"host_header\" cannot be set without also setting \"endpoint\"");
  }

  // For old-style buckets, default to us-east-1
  if (internal_kvstore_s3::ClassifyBucketName(bucket) ==
      internal_kvstore_s3::BucketNameType::kOldUSEast1) {
    if (!aws_region.empty() && aws_region != "us-east-1") {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Bucket ", QuoteString(bucket),
          " requires aws_region \"us-east-1\", not ", QuoteString(aws_region)));
    }
    aws_region = "us-east-1";
  }

  if (endpoint.empty()) {
    if (!aws_region.empty()) {
      if (!absl::StrContains(bucket, ".")) {
        // Use virtual host addressing.
        S3VirtualHostFormatter formatter;
        return S3EndpointRegion{
            formatter.GetEndpoint(bucket, aws_region),
            aws_region,
        };
      }

      // https://aws.amazon.com/blogs/aws/amazon-s3-path-deprecation-plan-the-rest-of-the-story/
      S3PathFormatter formatter;
      return S3EndpointRegion{
          formatter.GetEndpoint(bucket, aws_region),
          aws_region,
      };
    }

    return absl::OkStatus();
  }

  // Endpoint is specified.
  auto parsed = internal::ParseGenericUri(endpoint);
  if (parsed.scheme != "http" && parsed.scheme != "https") {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Endpoint ", endpoint, " has invalid scheme ",
                            parsed.scheme, ". Should be http(s)."));
  }
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Query in endpoint unsupported ", endpoint));
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Fragment in endpoint unsupported ", endpoint));
  }

  if (!aws_region.empty()) {
    // Endpoint and aws_region are specified; assume a non-virtual signing
    // header is allowed.
    S3CustomFormatter formatter{std::string(endpoint)};
    return S3EndpointRegion{
        formatter.GetEndpoint(bucket, aws_region),
        aws_region,
    };
  }

  return absl::OkStatus();
}

Future<S3EndpointRegion> ResolveEndpointRegion(
    std::string bucket, std::string_view endpoint, std::string host_header,
    std::shared_ptr<internal_http::HttpTransport> transport) {
  assert(!bucket.empty());
  assert(transport);
  assert(IsValidBucketName(bucket));

  if (endpoint.empty()) {
    // Use a HEAD request on bucket.s3.amazonaws.com to acquire the aws_region
    // does not work when there is a . in the bucket name; for now require the
    // aws_region to be set.
    if (!absl::StrContains(bucket, ".")) {
      std::string url = absl::StrFormat("https://%s.s3.amazonaws.com", bucket);
      return PromiseFuturePair<S3EndpointRegion>::Link(
                 ResolveHost<S3VirtualHostFormatter>{std::move(bucket),
                                                     S3VirtualHostFormatter{}},
                 transport->IssueRequest(
                     HttpRequestBuilder("HEAD", std::move(url))
                         .AddHostHeader(host_header)
                         .BuildRequest(),
                     {}))
          .future;
    }

    // The aws cli issues a request against the aws-global endpoint,
    // using host:s3.amazonaws.com, with a string-to-sign using "us-east-1"
    // zone. The response will be a 301 request with an 'x-amz-bucket-region'
    // header. We might be able to just do a signed HEAD request against an
    // possibly non-existent file... But try this later.
    std::string url =
        absl::StrFormat("https://s3.us-east-1.amazonaws.com/%s", bucket);
    return PromiseFuturePair<S3EndpointRegion>::Link(
               ResolveHost<S3PathFormatter>{std::move(bucket),
                                            S3PathFormatter{}},
               transport->IssueRequest(
                   HttpRequestBuilder("HEAD", std ::move(url))
                       .AddHostHeader(host_header)
                       .BuildRequest(),
                   {}))
        .future;
  }

  // Issue a HEAD request against the endpoint+bucket, which should work for
  // mock S3 backends like localstack or minio.
  std::string url = absl::StrFormat("%s/%s", endpoint, bucket);
  return PromiseFuturePair<S3EndpointRegion>::Link(
             ResolveHost<S3CustomFormatter>{
                 std::move(bucket), S3CustomFormatter{std::string(endpoint)}},
             transport->IssueRequest(HttpRequestBuilder("HEAD", std::move(url))
                                         .AddHostHeader(host_header)
                                         .BuildRequest(),
                                     {}))
      .future;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
