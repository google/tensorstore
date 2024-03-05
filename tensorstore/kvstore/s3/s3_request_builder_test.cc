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

#include "tensorstore/kvstore/s3/s3_request_builder.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/civil_time.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"

using ::tensorstore::internal_kvstore_s3::AwsCredentials;
using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;

namespace {

static const AwsCredentials credentials{
    "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", ""};
static const absl::TimeZone utc = absl::UTCTimeZone();
static constexpr char aws_region[] = "us-east-1";
static constexpr char bucket[] = "examplebucket";

TEST(S3RequestBuilderTest, SignatureMethods) {
  // https://github.com/aws/aws-sdk-cpp/blob/7be84d4a8d211dde517b9d9d9a1eca629072cce2/src/aws-cpp-sdk-core/source/http/URI.cpp#L30
  // The characters "$&,:=@" are legacy un-encoded characters and should be
  // encoded.
  // $ export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
  // $ export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  // $ aws s3 cp file.txt "s3://bucket/tensorstore/a-_.~$&,:=@z/b/file.txt"

  // NOTE: To get aws cli debug output for botocore & auth requires
  // intrusive changes to auth.py to enable DEBUG logging and avoid CRT auth, as
  // well as passing the flag --debug.

  /* CanonicalRequest:
PUT
/bucket/tensorstore/a-_.~%24%26%2C%3A%3D%40z/b/file.txt

content-md5:1B2M2Y8AsgTpgAmY7PhCfg==
content-type:text/plain
host:localhost:6678
x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
x-amz-date:20240221T030205Z

content-md5;content-type;host;x-amz-content-sha256;x-amz-date
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
*/

  /* StringToSign:
  AWS4-HMAC-SHA256
  20240221T030205Z
  20240221/us-west-2/s3/aws4_request
  e92fa24d7a44cb271a8ec0cbebcd09419ee64ab0c76c6f3563ab103298a44ac4
  */

  /* Signature:
  965b152573a180873c98bb6678ab50047fc24a06039de0874583298c7202bf0e
  */

  const auto now =
      absl::FromCivil(absl::CivilSecond(2024, 2, 21, 03, 02, 05), utc);

  auto builder =
      S3RequestBuilder(
          "PUT", "https://host/bucket/tensorstore/a-_.~$&,:=@z/b/file.txt")
          .AddHeader("content-md5: 1B2M2Y8AsgTpgAmY7PhCfg==")
          .AddHeader("content-type: text/plain");

  // NOTE: In a proper vhost request, `bucket/` would be omitted from the path.
  auto request = builder.BuildRequest(
      "bucket.s3.us-west-2.amazonaws.com", credentials, "us-west-2",
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", now);

  auto expected_canonical_request =
      "PUT\n"
      "/bucket/tensorstore/a-_.~%24%26%2C%3A%3D%40z/b/file.txt\n"
      "\n"
      "content-md5:1B2M2Y8AsgTpgAmY7PhCfg==\n"
      "content-type:text/plain\n"
      "host:bucket.s3.us-west-2.amazonaws.com\n"
      "x-amz-content-sha256:"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
      "x-amz-date:20240221T030205Z\n"
      "\n"
      "content-md5;content-type;host;x-amz-content-sha256;x-amz-date\n"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  auto expected_signing_string =
      "AWS4-HMAC-SHA256\n"
      "20240221T030205Z\n"
      "20240221/us-west-2/s3/aws4_request\n"
      "28c393b04c83956e1d4056351030e34bffa3dd877cf6cf2d0c83d2114bef7940";

  auto expected_signature =
      "c3bf762eae82b8a87dc5f7af8c2ad8973d4a0132c49bd8c46d025d4a1aa175fb";

  EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
  EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
  EXPECT_EQ(builder.GetSignature(), expected_signature);
}

TEST(S3RequestBuilderTest, AWS4SignatureGetExample) {
  // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
  // These values from worked exapmle in "Example: GET Object" Section
  auto url = absl::StrFormat("https://%s.s3.amazonaws.com/test.txt", bucket);
  auto builder = S3RequestBuilder("GET", url).AddHeader("range: bytes=0-9");
  auto request = builder.BuildRequest(
      "", credentials, aws_region,
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  auto expected_canonical_request =
      "GET\n"
      "/test.txt\n"
      "\n"
      "host:examplebucket.s3.amazonaws.com\n"
      "range:bytes=0-9\n"
      "x-amz-content-sha256:"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
      "x-amz-date:20130524T000000Z\n"
      "\n"
      "host;range;x-amz-content-sha256;x-amz-date\n"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  auto expected_signing_string =
      "AWS4-HMAC-SHA256\n"
      "20130524T000000Z\n"
      "20130524/us-east-1/s3/aws4_request\n"
      "7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972";

  auto expected_signature =
      "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";

  auto expected_auth_header =
      "Authorization: AWS4-HMAC-SHA256 "
      "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
      "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
      "Signature="
      "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";

  EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
  EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
  EXPECT_EQ(builder.GetSignature(), expected_signature);
  EXPECT_EQ(request.url, url);
  EXPECT_THAT(
      request.headers,
      ::testing::UnorderedElementsAre(
          expected_auth_header, "host: examplebucket.s3.amazonaws.com",
          "x-amz-content-sha256: "
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "x-amz-date: 20130524T000000Z", "range: bytes=0-9"));
}

TEST(S3RequestBuilderTest, AWS4SignaturePutExample) {
  // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
  // These values from worked example in "Example: PUT Object" Section
  auto url = absl::StrFormat("s3://%s/test$file.text", bucket);
  auto builder = S3RequestBuilder("PUT", url)
                     .AddHeader("date: Fri, 24 May 2013 00:00:00 GMT")
                     .AddHeader("x-amz-storage-class: REDUCED_REDUNDANCY");
  auto request = builder.BuildRequest(
      absl::StrFormat("%s.s3.amazonaws.com", bucket), credentials, aws_region,
      "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072",
      absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  auto expected_canonical_request =
      "PUT\n"
      "/test%24file.text\n"
      "\n"
      "date:Fri, 24 May 2013 00:00:00 GMT\n"
      "host:examplebucket.s3.amazonaws.com\n"
      "x-amz-content-sha256:"
      "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072\n"
      "x-amz-date:20130524T000000Z\n"
      "x-amz-storage-class:REDUCED_REDUNDANCY\n"
      "\n"
      "date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class\n"
      "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072";

  auto expected_signing_string =
      "AWS4-HMAC-SHA256\n"
      "20130524T000000Z\n"
      "20130524/us-east-1/s3/aws4_request\n"
      "9e0e90d9c76de8fa5b200d8c849cd5b8dc7a3be3951ddb7f6a76b4158342019d";

  auto expected_signature =
      "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";

  auto expected_auth_header =
      "Authorization: AWS4-HMAC-SHA256 "
      "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
      "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-"
      "class,"
      "Signature="
      "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";

  EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
  EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
  EXPECT_EQ(builder.GetSignature(), expected_signature);
  EXPECT_EQ(request.url, url);
  EXPECT_EQ(request.headers.size(), 6);
  EXPECT_THAT(
      request.headers,
      ::testing::UnorderedElementsAre(
          expected_auth_header, "date: Fri, 24 May 2013 00:00:00 GMT",
          "host: examplebucket.s3.amazonaws.com",
          "x-amz-content-sha256: "
          "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072",
          "x-amz-date: 20130524T000000Z",
          "x-amz-storage-class: REDUCED_REDUNDANCY"));
}

TEST(S3RequestBuilderTest, AWS4SignatureListObjectsExample) {
  // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
  // These values from worked example in "Example: GET Object" Section
  auto url = absl::StrFormat("https://%s.s3.amazonaws.com/", bucket);
  auto builder = S3RequestBuilder("GET", url)
                     .AddQueryParameter("prefix", "J")
                     .AddQueryParameter("max-keys", "2");
  auto request = builder.BuildRequest(
      absl::StrFormat("%s.s3.amazonaws.com", bucket), credentials, aws_region,
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  auto expected_canonical_request =
      "GET\n"
      "/\n"
      "max-keys=2&prefix=J\n"
      "host:examplebucket.s3.amazonaws.com\n"
      "x-amz-content-sha256:"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
      "x-amz-date:20130524T000000Z\n"
      "\n"
      "host;x-amz-content-sha256;x-amz-date\n"
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  auto expected_signing_string =
      "AWS4-HMAC-SHA256\n"
      "20130524T000000Z\n"
      "20130524/us-east-1/s3/aws4_request\n"
      "df57d21db20da04d7fa30298dd4488ba3a2b47ca3a489c74750e0f1e7df1b9b7";

  auto expected_signature =
      "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7";

  auto expected_auth_header =
      "Authorization: AWS4-HMAC-SHA256 "
      "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
      "SignedHeaders=host;x-amz-content-sha256;x-amz-date,"
      "Signature="
      "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7";

  EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
  EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
  EXPECT_EQ(builder.GetSignature(), expected_signature);
  EXPECT_EQ(request.url, absl::StrCat(url, "?max-keys=2&prefix=J"));
  EXPECT_EQ(request.headers.size(), 4);
  EXPECT_THAT(
      request.headers,
      ::testing::UnorderedElementsAre(
          expected_auth_header, "host: examplebucket.s3.amazonaws.com",
          "x-amz-content-sha256: "
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "x-amz-date: 20130524T000000Z"));
}

TEST(S3RequestBuilderTest, AnonymousCredentials) {
  // No Authorization header added for anonymous credentials
  auto url = absl::StrFormat("https://%s/test.txt", bucket);
  auto builder = S3RequestBuilder("GET", url).AddQueryParameter("test", "this");
  auto request = builder.BuildRequest(
      absl::StrFormat("%s.s3.amazonaws.com", bucket), AwsCredentials{},
      aws_region,
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  EXPECT_EQ(request.url, absl::StrCat(url, "?test=this"));
  EXPECT_EQ(request.headers.size(), 3);
  EXPECT_THAT(request.headers, ::testing::Not(::testing::Contains(
                                   ::testing::HasSubstr("Authorization:"))));
  EXPECT_THAT(
      request.headers,
      ::testing::UnorderedElementsAre(
          "host: examplebucket.s3.amazonaws.com",
          "x-amz-content-sha256: "
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "x-amz-date: 20130524T000000Z"));
}

TEST(S3RequestBuilderTest, AwsSessionTokenHeaderAdded) {
  /// Only test that x-amz-security-token is added if present on AwsCredentials
  auto token = "abcdef1234567890";
  auto sts_credentials =
      AwsCredentials{credentials.access_key, credentials.secret_key, token};
  auto builder =
      S3RequestBuilder("GET", absl::StrFormat("https://%s/test.txt", bucket));
  auto request = builder.BuildRequest(
      absl::StrFormat("%s.s3.amazonaws.com", bucket), sts_credentials,
      aws_region,
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  EXPECT_EQ(request.headers.size(), 5);
  EXPECT_THAT(request.headers,
              ::testing::Contains(::testing::HasSubstr("Authorization: ")));
  EXPECT_THAT(request.headers, ::testing::Contains(absl::StrCat(
                                   "x-amz-security-token: ", token)));
}

TEST(S3RequestBuilderTest, AwsRequesterPaysHeaderAdded) {
  /// Test that x-amz-requester-payer: requester is added if true
  auto request =
      S3RequestBuilder("GET", absl::StrFormat("https://%s/test.txt", bucket))
          .MaybeAddRequesterPayer(false)
          .BuildRequest(
              absl::StrFormat("%s.s3.amazonaws.com", bucket), credentials,
              aws_region,
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"
              "5",
              absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  EXPECT_THAT(request.headers,
              ::testing::Not(::testing::Contains(
                  ::testing::HasSubstr("x-amz-requester-payer"))));

  request =
      S3RequestBuilder("GET", absl::StrFormat("https://%s/test.txt", bucket))
          .MaybeAddRequesterPayer(true)
          .BuildRequest(
              absl::StrFormat("%s.s3.amazonaws.com", bucket), credentials,
              aws_region,
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"
              "5",
              absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

  EXPECT_THAT(request.headers,
              ::testing::Contains("x-amz-requester-payer: requester"));
}

}  // namespace
