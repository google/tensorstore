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
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/mock_http_transport.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_http::DefaultMockHttpTransport;
using ::tensorstore::internal_http::HeaderMap;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_kvstore_s3::ConditionalWriteMode;
using ::tensorstore::internal_kvstore_s3::IsAwsS3Endpoint;
using ::tensorstore::internal_kvstore_s3::ResolveEndpointRegion;
using ::tensorstore::internal_kvstore_s3::S3EndpointRegion;
using ::tensorstore::internal_kvstore_s3::ValidateEndpoint;

namespace {

TEST(IsAwsS3Endpoint, PositiveExamples) {
  for (
      std::string_view uri : {
          "http://s3.amazonaws.com",         //
          "http://s3.amazonaws.com/bucket",  //
          "http://bucket.s3.amazonaws.com/path",
          "https://bucket.s3.amazonaws.com/path",
          // https://docs.aws.amazon.com/general/latest/gr/s3.html
          "http://s3.us-west-2.amazonaws.com",
          "http://s3.dualstack.us-west-2.amazonaws.com",
          "http://s3-fips.dualstack.us-west-2.amazonaws.com",
          "http://s3-fips.us-west-2.amazonaws.com",
          "http://s3.us-west-2.amazonaws.com/bucket",
          "http://s3.dualstack.us-west-2.amazonaws.com/bucket",
          "http://s3-fips.dualstack.us-west-2.amazonaws.com/bucket",
          "http://s3-fips.us-west-2.amazonaws.com/bucket",
          "http://foo.bar.com.s3.us-west-2.amazonaws.com",
          "http://foo.bar.com.s3.dualstack.us-west-2.amazonaws.com",
          "http://foo.bar.com.s3-fips.dualstack.us-west-2.amazonaws.com",
          "http://foo.bar.com.s3-fips.us-west-2.amazonaws.com",
          // gov-cloud examples
          "http://s3.us-gov-west-1.amazonaws.com",
          "http://s3-fips.dualstack.us-gov-west-1.amazonaws.com",
          "http://s3.dualstack.us-gov-west-1.amazonaws.com",
          "http://s3-fips.us-gov-west-1.amazonaws.com",
          "http://s3.us-gov-west-1.amazonaws.com/bucket",
          "http://s3-fips.dualstack.us-gov-west-1.amazonaws.com/bucket",
          "http://s3.dualstack.us-gov-west-1.amazonaws.com/bucket",
          "http://s3-fips.us-gov-west-1.amazonaws.com/bucket",
          "http://foo-bar-com.s3.us-gov-west-1.amazonaws.com",
          "http://foo-bar-com.s3-fips.dualstack.us-gov-west-1.amazonaws.com",
          "http://foo-bar-com.s3.dualstack.us-gov-west-1.amazonaws.com",
          "http://foo-bar-com.s3-fips.us-gov-west-1.amazonaws.com",
          // https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html
          "http://amzn-s3-demo-bucket1.s3.eu-west-1.amazonaws.com/",
          // The s3 bucket looks like the s3 amazonaws host name.
          "http://s3.fakehoster.com.s3.us-west-2.amazonaws.com",
      }) {
    EXPECT_TRUE(IsAwsS3Endpoint(uri)) << uri;
  }
}

TEST(IsAwsS3Endpoint, NegativeExamples) {
  for (std::string_view uri : {
           "http://localhost:1234/path",
           "http://bard.amazonaws.com/bucket",         //
           "https://s3.fakehoster.com.amazonaws.com",  //
       }) {
    EXPECT_FALSE(IsAwsS3Endpoint(uri)) << uri;
  }
}

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
                  S3EndpointRegion{"http://my.host/testbucket", "region",
                                   ConditionalWriteMode::kDefault}));

  EXPECT_THAT(
      ValidateEndpoint("testbucket", "region", "http://my.host", "my.header"),
      ::testing::VariantWith<S3EndpointRegion>(
          S3EndpointRegion{"http://my.host/testbucket", "region",
                           ConditionalWriteMode::kDefault}));

  // error: host header with empty endpoint
  EXPECT_THAT(ValidateEndpoint("testbucket", {}, {}, "my.header"),
              ::testing::VariantWith<absl::Status>(
                  tensorstore::StatusIs(absl::StatusCode::kInvalidArgument)));
}

// Mock-based tests for s3.

TEST(ResolveEndpointRegion, Basic) {
  auto mock_transport = std::make_shared<DefaultMockHttpTransport>(
      DefaultMockHttpTransport::Responses{
          // initial HEAD request responds with an x-amz-bucket-region header.
          {"HEAD https://testbucket.s3.amazonaws.com",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"HEAD https://s3.us-east-1.amazonaws.com/test.bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          {"HEAD http://localhost:1234/test.bucket",
           HttpResponse{200, absl::Cord(),
                        HeaderMap{{"x-amz-bucket-region", "us-east-1"}}}},

          // x-amz-request-id ends with -ceph3
          {"HEAD http://localhost.ceph/test.bucket",
           HttpResponse{
               200, absl::Cord(),
               HeaderMap{{"x-amz-bucket-region", "us-east-1"},
                         {"x-amz-request-id", "tx000001abcdef-prod1-ceph3"}}}},
          // DELETE 404 => absl::OkStatus()
      });

  S3EndpointRegion ehr;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr,
      ResolveEndpointRegion("testbucket", {}, {}, mock_transport).result());

  EXPECT_THAT(ehr.endpoint, "https://testbucket.s3.us-east-1.amazonaws.com");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
  EXPECT_THAT(ehr.write_mode, ConditionalWriteMode::kEnabled);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr,
      ResolveEndpointRegion("test.bucket", {}, {}, mock_transport).result());

  EXPECT_THAT(ehr.endpoint, "https://s3.us-east-1.amazonaws.com/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
  EXPECT_THAT(ehr.write_mode, ConditionalWriteMode::kEnabled);

  // With endpoint
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr, ResolveEndpointRegion("test.bucket", "http://localhost:1234", {},
                                 mock_transport)
               .result());

  EXPECT_THAT(ehr.endpoint, "http://localhost:1234/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
  EXPECT_THAT(ehr.write_mode, ConditionalWriteMode::kDefault);

  // With endpoint & host_header
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr, ResolveEndpointRegion("test.bucket", "http://localhost:1234",
                                 "s3.localhost.com", mock_transport)
               .result());

  EXPECT_THAT(ehr.endpoint, "http://localhost:1234/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
  EXPECT_THAT(ehr.write_mode, ConditionalWriteMode::kDefault);

  // With ceph endpoint
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      ehr, ResolveEndpointRegion("test.bucket", "http://localhost.ceph", {},
                                 mock_transport)
               .result());

  EXPECT_THAT(ehr.endpoint, "http://localhost.ceph/test.bucket");
  EXPECT_THAT(ehr.aws_region, "us-east-1");
  EXPECT_THAT(ehr.write_mode, ConditionalWriteMode::kDisabled);
}

}  // namespace
