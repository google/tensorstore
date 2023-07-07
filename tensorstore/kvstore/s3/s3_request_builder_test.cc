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
#include "tensorstore/kvstore/s3/s3_request_builder.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_storage_s3::S3RequestBuilder;
using ::tensorstore::internal_http::HttpRequest;

namespace {


TEST(S3RequestBuilderTest, AWS4SignatureGetExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: GET Object" Section
    auto cs = absl::CivilSecond(2013, 5, 24, 0, 0, 0);
    auto utc = absl::UTCTimeZone();
    auto time = absl::FromCivil(cs, utc);

    auto aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    auto aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    auto aws_region = "us-east-1";
    auto bucket = "examplebucket";
    auto payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    auto url = absl::StrFormat("https://%s/test.txt", bucket);
    auto host = absl::StrFormat("%s.s3.amazonaws.com", bucket);
    auto x_amz_date = absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc);

    std::vector<std::pair<std::string, std::string>> headers = {
        {"host", host},
        {"range", "bytes=0-9"},
        {"x-amz-content-sha256", payload_hash},
        {"x-amz-date", x_amz_date}
    };

    auto canonical_request = S3RequestBuilder::CanonicalRequest(url, "GET", payload_hash, headers, {});

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

    EXPECT_EQ(canonical_request, expected_canonical_request);

    auto signing_string = S3RequestBuilder::SigningString(canonical_request, aws_region, time);

    auto expected_signing_string =
        "AWS4-HMAC-SHA256\n"
        "20130524T000000Z\n"
        "20130524/us-east-1/s3/aws4_request\n"
        "7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972";

    EXPECT_EQ(signing_string, expected_signing_string);

    auto expected_signature = "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";
    auto signature = S3RequestBuilder::Signature(aws_secret_access_key, aws_region, signing_string, time);
    EXPECT_EQ(signature, expected_signature);

    auto expected_auth_header =
        "Authorization: AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";

    auto auth_header = S3RequestBuilder::AuthorizationHeader(aws_access_key, aws_region, signature, headers, time);
    EXPECT_EQ(auth_header, expected_auth_header);

    auto s3_builder = S3RequestBuilder("GET", url);
    for(auto it = headers.rbegin(); it != headers.rend(); ++it) {
        s3_builder.AddHeader(absl::StrCat(it->first, ": ", it->second));
    }
    auto request = s3_builder.BuildRequest(aws_access_key, aws_secret_access_key,
                                           aws_region, payload_hash, time);

    EXPECT_THAT(request.headers, ::testing::Contains(auth_header));

}


TEST(S3RequestBuilderTest, AWS4SignaturePutExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: PUT Object" Section
    auto cs = absl::CivilSecond(2013, 5, 24, 0, 0, 0);
    auto utc = absl::UTCTimeZone();
    auto time = absl::FromCivil(cs, utc);

    auto aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    auto aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    auto aws_region = "us-east-1";
    auto bucket = "examplebucket";
    auto payload_hash = "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072";

    auto url = absl::StrFormat("s3://%s/test$file.text", bucket);
    auto host = absl::StrFormat("%s.s3.amazonaws.com", bucket);
    auto x_amz_date = absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc);

    std::vector<std::pair<std::string, std::string>> headers = {
        {"date", "Fri, 24 May 2013 00:00:00 GMT"},
        {"host", host},
        {"x-amz-content-sha256", payload_hash},
        {"x-amz-date", x_amz_date},
        {"x-amz-storage-class", "REDUCED_REDUNDANCY"}
    };

    auto canonical_request = S3RequestBuilder::CanonicalRequest(url, "PUT", payload_hash, headers, {});

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


    EXPECT_EQ(canonical_request, expected_canonical_request);

    auto signing_string = S3RequestBuilder::SigningString(canonical_request, aws_region, time);
    auto expected_signing_string =
        "AWS4-HMAC-SHA256\n"
        "20130524T000000Z\n"
        "20130524/us-east-1/s3/aws4_request\n"
        "9e0e90d9c76de8fa5b200d8c849cd5b8dc7a3be3951ddb7f6a76b4158342019d";
    EXPECT_EQ(signing_string, expected_signing_string);

    auto expected_signature = "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";
    auto signature = S3RequestBuilder::Signature(aws_secret_access_key, aws_region, signing_string, time);
    EXPECT_EQ(signature, expected_signature);

    auto expected_auth_header =
        "Authorization: AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class,"
        "Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";

    auto auth_header = S3RequestBuilder::AuthorizationHeader(aws_access_key, aws_region, signature, headers, time);
    EXPECT_EQ(auth_header, expected_auth_header);

    auto s3_builder = S3RequestBuilder("PUT", url);
    for(auto it = headers.rbegin(); it != headers.rend(); ++it) {
        s3_builder.AddHeader(absl::StrCat(it->first, ": ", it->second));
    }
    auto request = s3_builder.BuildRequest(aws_access_key, aws_secret_access_key,
                                           aws_region, payload_hash, time);

    EXPECT_THAT(request.headers, ::testing::Contains(auth_header));
}


TEST(S3RequestBuilderTest, AWS4SignatureListObjectsExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: GET Object" Section
    auto cs = absl::CivilSecond(2013, 5, 24, 0, 0, 0);
    auto utc = absl::UTCTimeZone();
    auto time = absl::FromCivil(cs, utc);

    auto aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    auto aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    auto aws_region = "us-east-1";
    auto bucket = "examplebucket";
    auto payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    auto url = absl::StrFormat("https://%s/", bucket);
    auto host = absl::StrFormat("%s.s3.amazonaws.com", bucket);
    auto x_amz_date = absl::FormatTime("%Y%m%dT%H%M%SZ", time, utc);

    std::vector<std::pair<std::string, std::string>> headers = {
        {"host", host},
        {"x-amz-content-sha256", payload_hash},
        {"x-amz-date", x_amz_date}
    };

    std::vector<std::pair<std::string, std::string>> queries = {
        { "max-keys", "2" },
        { "prefix", "J"},
    };

    auto canonical_request = S3RequestBuilder::CanonicalRequest(url, "GET", payload_hash, headers, queries);

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

    EXPECT_EQ(canonical_request, expected_canonical_request);

    auto signing_string = S3RequestBuilder::SigningString(canonical_request, aws_region, time);

    auto expected_signing_string =
        "AWS4-HMAC-SHA256\n"
        "20130524T000000Z\n"
        "20130524/us-east-1/s3/aws4_request\n"
        "df57d21db20da04d7fa30298dd4488ba3a2b47ca3a489c74750e0f1e7df1b9b7";

    EXPECT_EQ(signing_string, expected_signing_string);

    auto expected_signature = "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7";
    auto signature = S3RequestBuilder::Signature(aws_secret_access_key, aws_region, signing_string, time);
    EXPECT_EQ(signature, expected_signature);

    auto expected_auth_header =
        "Authorization: AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=host;x-amz-content-sha256;x-amz-date,"
        "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7";

    auto auth_header = S3RequestBuilder::AuthorizationHeader(aws_access_key, aws_region, signature, headers, time);
    EXPECT_EQ(auth_header, expected_auth_header);

    auto s3_builder = S3RequestBuilder("GET", url);
    for(auto it = headers.rbegin(); it != headers.rend(); ++it) {
        s3_builder.AddHeader(absl::StrCat(it->first, ": ", it->second));
    }
    for(auto it = queries.rbegin(); it != queries.rend(); ++it) s3_builder.AddQueryParameter(it->first, it->second);
    auto request = s3_builder.BuildRequest(aws_access_key, aws_secret_access_key,
                                           aws_region, payload_hash, time);

    EXPECT_THAT(request.headers, ::testing::Contains(auth_header));
}

} // namespace
