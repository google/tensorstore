// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_S3_AWS_HTTP_MOCKING_H_
#define TENSORSTORE_KVSTORE_S3_AWS_HTTP_MOCKING_H_

#include <string>
#include <utility>
#include <vector>

#include <aws/auth/credentials.h>
#include "tensorstore/internal/http/http_response.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

aws_auth_http_system_vtable* GetAwsHttpMockingIfEnabled();

using AwsHttpMockingResponses =
    std::vector<std::pair<std::string, internal_http::HttpResponse>>;

/// Enables mocking of AWS HTTP requests.
///
/// The first matching pair will be returned for each call, then expired.
void EnableAwsHttpMocking(AwsHttpMockingResponses responses);

/// Disables mocking of AWS HTTP requests.
void DisableAwsHttpMocking();

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_AWS_HTTP_MOCKING_H_
