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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/mock_http_transport.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_http::DefaultMockHttpTransport;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::ResolveEndpointRegion;
using ::tensorstore::internal_kvstore_s3::S3EndpointRegion;
using ::tensorstore::internal_kvstore_s3::ValidateEndpoint;

namespace {

TEST(ValidateEndpointTest, Basic) {
  // {bucket} => Ok (must be resolved later)
  EXPECT_THAT(ValidateEndpoint("testbucket", {}, {}, {}),
              ::testing::VariantWith<absl::Status>(absl::OkStatus()));

  EXPECT_THAT(ValidateEndpoint("test.bucket", {}, {}, {}),
              ::testing::VariantWith<absl::Status>(absl::OkStatus()));

  // {bucket, region} => Immediately resolved.
  EXPECT_THAT(ValidateEndpoint("testbucket", "us-east-1", {}, {}),
              ::testing::VariantWith<S3EndpointRegion>(testing::_));

  // kOldUSEast1 bucket
  EXPECT_THAT(ValidateEndpoint("OldBucket", "us-east-1", {}, {}),
              ::testing::VariantWith<S3EndpointRegion>(testing::_));

  EXPECT_THAT(ValidateEndpoint("OldBucket", {}, {}, {}),
              ::testing::VariantWith<S3EndpointRegion>(testing::_));

  // error: kOldUSEast1 bucket not in us-east-1
  EXPECT_THAT(ValidateEndpoint("OldBucket", "us-west-1", {}, {}),
              ::testing::VariantWith<absl::Status>(
                  tensorstore::StatusIs(absl::StatusCode::kInvalidArgument)));

  EXPECT_THAT(ValidateEndpoint("testbucket", "region", "http://my.host", {}),
              ::testing::VariantWith<S3EndpointRegion>(
                  S3EndpointRegion{"http://my.host/testbucket", "region"}));

  EXPECT_THAT(
      ValidateEndpoint("testbucket", "region", "http://my.host", "my.header"),
      ::testing::VariantWith<S3EndpointRegion>(
          S3EndpointRegion{"http://my.host/testbucket", "region"}));

  // error: host header with empty endpoint
  EXPECT_THAT(ValidateEndpoint("testbucket", {}, {}, "my.header"),
              ::testing::VariantWith<absl::Status>(
                  tensorstore::StatusIs(absl::StatusCode::kInvalidArgument)));
}

// Mock-based tests for s3.

TEST(ResolveEndpointRegion, Basic) {
  absl::flat_hash_map<std::string, HttpResponse> url_to_response{
      // initial HEAD request responds with an x-amz-bucket-region header.
      {"HEAD https://testbucket.s3.amazonaws.com",
       HttpResponse{200, absl::Cord(), {{"x-amz-bucket-region", "us-east-1"}}}},

      {"HEAD https://s3.us-east-1.amazonaws.com/test.bucket",
       HttpResponse{200, absl::Cord(), {{"x-amz-bucket-region", "us-east-1"}}}},

      {"HEAD http://localhost:1234/test.bucket",
       HttpResponse{200, absl::Cord(), {{"x-amz-bucket-region", "us-east-1"}}}},

      // DELETE 404 => absl::OkStatus()
  };

  auto mock_transport =
      std::make_shared<DefaultMockHttpTransport>(url_to_response);
  S3EndpointRegion ehr;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr,
      ResolveEndpointRegion("testbucket", {}, {}, mock_transport).result());

  EXPECT_THAT(ehr.endpoint, "https://testbucket.s3.us-east-1.amazonaws.com");
  EXPECT_THAT(ehr.aws_region, "us-east-1");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr,
      ResolveEndpointRegion("test.bucket", {}, {}, mock_transport).result());

  EXPECT_THAT(ehr.endpoint, "https://s3.us-east-1.amazonaws.com/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");

  // With endpoint
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr, ResolveEndpointRegion("test.bucket", "http://localhost:1234", {},
                                 mock_transport)
               .result());

  EXPECT_THAT(ehr.endpoint, "http://localhost:1234/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");

  // With endpoint & host_header
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr, ResolveEndpointRegion("test.bucket", "http://localhost:1234",
                                 "s3.localhost.com", mock_transport)
               .result());

  EXPECT_THAT(ehr.endpoint, "http://localhost:1234/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
}

}  // namespace
