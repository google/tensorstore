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

#ifndef TENSORSTORE_KVSTORE_S3_S3_ENDPOINT_H_
#define TENSORSTORE_KVSTORE_S3_S3_ENDPOINT_H_

#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Resolved S3 endpoint, host header, and aws_regions.
///  This is used to issue
///. re
///
struct S3EndpointRegion {
  /// The base http-endpoint used to access an S3 bucket. This is a url and
  /// optional path prefix for accessing a bucket.
  std::string endpoint;

  /// The AWS region for the S3 bucket, which is required for constructing a
  /// request signature.
  std::string aws_region;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const S3EndpointRegion& ehr) {
    absl::Format(&sink, "S3EndpointRegion{endpoint=%s, aws_region=%s}",
                 ehr.endpoint, ehr.aws_region);
  }

  friend bool operator==(const S3EndpointRegion& a, const S3EndpointRegion& b) {
    return a.endpoint == b.endpoint && a.aws_region == b.aws_region;
  }

  friend bool operator!=(const S3EndpointRegion& a, const S3EndpointRegion& b) {
    return !(a == b);
  }
};

/// Validate the bucket, endpoint and host_header parameters.
/// When possible, constructs an S3EndpointRegion from the driver config.
///
/// Returns an absl::Status or an S3EndpointRegion.
/// When the return value holds:
/// * An error status: The validation failed.
/// * An OK status: The endpoint could not be immediately resolved, and
///   ResolveEndpointRegion should be called.
/// * An `S3EndpointRegion`: Endpoint is fully resolved without additional
///   requests.
std::variant<absl::Status, S3EndpointRegion> ValidateEndpoint(
    std::string_view bucket, std::string aws_region, std::string_view endpoint,
    std::string host_header);

/// Resolve an endpoint to fill a S3EndpointRegion.
/// This issues a request to AWS to determine the proper endpoint, host_header,
/// and aws_region for a given bucket.
/// Pre:
///   ValidateEndpoint has returned an `absl::OkStatus`.
Future<S3EndpointRegion> ResolveEndpointRegion(
    std::string bucket, std::string_view endpoint, std::string host_header,
    std::shared_ptr<internal_http::HttpTransport> transport);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_S3_ENDPOINT_H_
