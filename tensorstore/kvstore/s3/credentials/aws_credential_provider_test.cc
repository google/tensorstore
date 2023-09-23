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

#include "tensorstore/kvstore/s3/credentials/aws_credential_provider.h"

#include <fstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::internal::GetEnv;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::HttpTransport;
using ::tensorstore::internal_kvstore_s3::GetAwsCredentialProvider;


/// Cause EC2Metadata to always fail
class NotFoundTransport : public HttpTransport {
public:
  Future<HttpResponse> IssueRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout) override {
    return HttpResponse{404, absl::Cord(), {}};
  }
};


class AwsCredentialProviderTest : public ::testing::Test {
 protected:
  std::shared_ptr<NotFoundTransport> transport_;

  void SetUp() override {
    // Make sure that env vars are not set.
    for (const char* var :
         {"AWS_SHARED_CREDENTIALS_FILE", "AWS_ACCESS_KEY_ID",
          "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"}) {
      UnsetEnv(var);
    }

    transport_ = std::make_shared<NotFoundTransport>();
  }
};

}  // namespace
