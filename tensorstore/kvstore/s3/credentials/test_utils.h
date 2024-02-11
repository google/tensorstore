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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_TEST_UTILS_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_TEST_UTILS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Mocks an HttpTransport by overriding the IssueRequest method to
/// respond with a predefined set of request-response pairs supplied
/// to the constructor
class EC2MetadataMockTransport : public internal_http::HttpTransport {
 public:
  EC2MetadataMockTransport(
      const absl::flat_hash_map<std::string, internal_http::HttpResponse>&
          url_to_response)
      : url_to_response_(url_to_response) {}

  Future<internal_http::HttpResponse> IssueRequest(
      const internal_http::HttpRequest& request, absl::Cord payload,
      absl::Duration request_timeout, absl::Duration connect_timeout) override;

  const absl::flat_hash_map<std::string, internal_http::HttpResponse>&
      url_to_response_;
};

/// Return a Default EC2 Metadata Credential Retrieval Flow, suitable
/// for passing to EC2MetadataMockTransport
absl::flat_hash_map<std::string, internal_http::HttpResponse>
DefaultEC2MetadataFlow(const std::string& endpoint,
                       const std::string& api_token,
                       const std::string& access_key,
                       const std::string& secret_key,
                       const std::string& session_token,
                       const absl::Time& expires_at);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_TEST_UTILS_H_
