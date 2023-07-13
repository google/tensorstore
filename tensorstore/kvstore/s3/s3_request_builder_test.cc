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

#include <iostream>

#include <gtest/gtest.h>

#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/s3/s3_credential_provider.h"
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_kvstore_s3::S3RequestBuilder;
using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_kvstore_s3::S3Credentials;

namespace {

static const S3Credentials credentials{"AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", ""};
static const absl::TimeZone utc = absl::UTCTimeZone();
static constexpr char aws_region[] = "us-east-1";
static constexpr char bucket[] = "examplebucket";

TEST(S3RequestBuilderTest, AWS4SignatureGetExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: GET Object" Section
    auto url = absl::StrFormat("https://%s/test.txt", bucket);
    auto builder = S3RequestBuilder("GET", url)
                        .AddHeader("range: bytes=0-9");
    auto request = builder.BuildRequest(
        absl::StrFormat("%s.s3.amazonaws.com", bucket),
        credentials, aws_region,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

    auto expected_canonical_request =
        "GET\n"
        "/test.txt\n"
        "\n"
        "host:examplebucket.s3.amazonaws.com\n"
        "range:bytes=0-9\n"
        "x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
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
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";

    EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
    EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
    EXPECT_EQ(builder.GetSignature(), expected_signature);
    EXPECT_EQ(request.url, url);
    EXPECT_EQ(request.headers.size(), 5);
    EXPECT_THAT(request.headers, ::testing::Contains(expected_auth_header));
    EXPECT_THAT(request.headers, ::testing::Contains("host: examplebucket.s3.amazonaws.com"));
    EXPECT_THAT(request.headers, ::testing::Contains(
        "x-amz-content-sha256: "
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    EXPECT_THAT(request.headers, ::testing::Contains("x-amz-date: 20130524T000000Z"));
    EXPECT_THAT(request.headers, ::testing::Contains("range: bytes=0-9"));
}


TEST(S3RequestBuilderTest, AWS4SignaturePutExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked example in "Example: PUT Object" Section
    auto url = absl::StrFormat("s3://%s/test$file.text", bucket);
    auto builder = S3RequestBuilder("PUT", url)
                        .AddHeader("date: Fri, 24 May 2013 00:00:00 GMT")
                        .AddHeader("x-amz-storage-class: REDUCED_REDUNDANCY");
    auto request = builder.BuildRequest(
        absl::StrFormat("%s.s3.amazonaws.com", bucket),
        credentials, aws_region,
        "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072",
        absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

    auto expected_canonical_request =
        "PUT\n"
        "/test%24file.text\n"
        "\n"
        "date:Fri, 24 May 2013 00:00:00 GMT\n"
        "host:examplebucket.s3.amazonaws.com\n"
        "x-amz-content-sha256:44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072\n"
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
        "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class,"
        "Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";

    EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
    EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
    EXPECT_EQ(builder.GetSignature(), expected_signature);
    EXPECT_EQ(request.url, url);
    EXPECT_EQ(request.headers.size(), 6);
    EXPECT_THAT(request.headers, ::testing::Contains(expected_auth_header));
    EXPECT_THAT(request.headers, ::testing::Contains("date: Fri, 24 May 2013 00:00:00 GMT"));
    EXPECT_THAT(request.headers, ::testing::Contains("host: examplebucket.s3.amazonaws.com"));
    EXPECT_THAT(request.headers, ::testing::Contains(
        "x-amz-content-sha256: "
        "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"));
    EXPECT_THAT(request.headers, ::testing::Contains("x-amz-date: 20130524T000000Z"));
    EXPECT_THAT(request.headers, ::testing::Contains("x-amz-storage-class: REDUCED_REDUNDANCY"));

}


TEST(S3RequestBuilderTest, AWS4SignatureListObjectsExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked example in "Example: GET Object" Section
    auto url = absl::StrFormat("https://%s/", bucket);
    auto builder = S3RequestBuilder("GET", url)
                        .AddQueryParameter("prefix", "J")
                        .AddQueryParameter("max-keys", "2");
    auto request = builder.BuildRequest(
        absl::StrFormat("%s.s3.amazonaws.com", bucket),
        credentials, aws_region,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

    auto expected_canonical_request =
        "GET\n"
        "/\n"
        "max-keys=2&prefix=J\n"
        "host:examplebucket.s3.amazonaws.com\n"
        "x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
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
        "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7";

    EXPECT_EQ(builder.GetCanonicalRequest(), expected_canonical_request);
    EXPECT_EQ(builder.GetSigningString(), expected_signing_string);
    EXPECT_EQ(builder.GetSignature(), expected_signature);
    EXPECT_EQ(request.url, absl::StrCat(url, "?max-keys=2&prefix=J"));
    EXPECT_EQ(request.headers.size(), 4);
    EXPECT_THAT(request.headers, ::testing::Contains(expected_auth_header));
    EXPECT_THAT(request.headers, ::testing::Contains("host: examplebucket.s3.amazonaws.com"));
    EXPECT_THAT(request.headers, ::testing::Contains(
        "x-amz-content-sha256: "
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    EXPECT_THAT(request.headers, ::testing::Contains("x-amz-date: 20130524T000000Z"));
}

TEST(S3RequestBuilderTest, AnonymousCredentials) {
    // No Authorization header added for anonymous credentials
    auto url = absl::StrFormat("https://%s/test.txt", bucket);
    auto builder = S3RequestBuilder("GET", url).AddQueryParameter("test", "this");
    auto request = builder.BuildRequest(
        absl::StrFormat("%s.s3.amazonaws.com", bucket),
        S3Credentials{}, aws_region,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

    EXPECT_EQ(request.url, absl::StrCat(url, "?test=this"));
    EXPECT_EQ(request.headers.size(), 3);
    EXPECT_THAT(request.headers, ::testing::Not(
            ::testing::Contains(::testing::HasSubstr("Authorization:"))));
    EXPECT_THAT(request.headers, ::testing::Contains("host: examplebucket.s3.amazonaws.com"));
    EXPECT_THAT(request.headers, ::testing::Contains(
        "x-amz-content-sha256: "
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    EXPECT_THAT(request.headers, ::testing::Contains("x-amz-date: 20130524T000000Z"));
}

TEST(S3RequestBuilderTest, AwsSessionTokenHeaderAdded) {
    /// Only test that x-amz-security-token is added if present on S3Credentials
    auto token = "abcdef1234567890";
    auto sts_credentials = S3Credentials{credentials.access_key, credentials.secret_key, token};
    auto builder = S3RequestBuilder("GET", absl::StrFormat("https://%s/test.txt", bucket));
    auto request = builder.BuildRequest(
        absl::StrFormat("%s.s3.amazonaws.com", bucket),
        sts_credentials, aws_region,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        absl::FromCivil(absl::CivilSecond(2013, 5, 24, 0, 0, 0), utc));

    EXPECT_EQ(request.headers.size(), 5);
    EXPECT_THAT(request.headers, ::testing::Contains(::testing::HasSubstr("Authorization: ")));
    EXPECT_THAT(request.headers, ::testing::Contains(absl::StrCat("x-amz-security-token: ", token)));
}

} // namespace
