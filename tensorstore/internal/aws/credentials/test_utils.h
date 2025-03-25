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
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "tensorstore/internal/http/http_response.h"

namespace tensorstore {
namespace internal_aws {

/// Return a Default EC2 Metadata Credential Retrieval Flow, suitable
/// for passing to EC2MetadataMockTransport
std::vector<std::pair<std::string, internal_http::HttpResponse>>
DefaultImdsCredentialFlow(const std::string& api_token,
                          const std::string& access_key,
                          const std::string& secret_key,
                          const std::string& session_token,
                          const absl::Time& expires_at);

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_TEST_UTILS_H_
