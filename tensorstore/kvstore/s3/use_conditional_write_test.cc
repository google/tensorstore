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

#include "tensorstore/kvstore/s3/use_conditional_write.h"

#include <string_view>

#include <gtest/gtest.h>

using ::tensorstore::internal_kvstore_s3::IsAwsS3Endpoint;

namespace {

TEST(IsAwsS3Endpoint, PositiveExamples) {
  for (
      std::string_view uri : {
          "http://s3.amazonaws.com",         //
          "http://s3.amazonaws.com/bucket",  //
          "http://bucket.s3.amazonaws.com/path",
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
           "http://s3.fakehoster.com.amazonaws.com",  //
           "http://bard.amazonaws.com/bucket",        //
       }) {
    EXPECT_FALSE(IsAwsS3Endpoint(uri)) << uri;
  }
}

}  // namespace
