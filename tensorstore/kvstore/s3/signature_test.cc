// Copyright 2020 The TensorStore Authors
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
#include "tensorstore/kvstore/s3/signature.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/status_testutil.h"


using ::tensorstore::internal::ParseGenericUri;
using ::tensorstore::internal::ParsedGenericUri;
using ::tensorstore::internal_storage_s3::CanonicalRequest;
using ::tensorstore::internal_storage_s3::SigningString;
using ::tensorstore::internal_storage_s3::Signature;

namespace {

TEST(S3KeyValueStoreTest, AWSSignatureGetExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: GET Object" Section
    absl::CivilSecond cs(2013, 5, 24, 0, 0, 0);
    auto utc = absl::UTCTimeZone();
    auto time = absl::FromCivil(cs, utc);

    std::string aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    std::string aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    std::string aws_region = "us-east-1";
    std::string bucket = "examplebucket";

    auto str_uri = absl::StrFormat("s3://%s/test.txt", bucket);
    auto uri = ParseGenericUri(str_uri);
    auto host = absl::StrFormat("%s.s3.amazonaws.com", bucket);
    auto x_amz_date = absl::FormatTime("%Y%m%dT%H%M%SZ\n", time, utc);

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto canonical_request,
        CanonicalRequest(
            "GET",
            uri,
            {
                { "Host", host },
                { "range", "bytes=0-9" },
                { "x-amz-content-sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
                { "x-amz-date", x_amz_date },
            },
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    );

    std::string expected_canonical_request =
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

    auto signing_string = SigningString(canonical_request, time, aws_region);

    std::string expected_signing_string =
        "AWS4-HMAC-SHA256\n" +
        absl::FormatTime("%Y%m%dT%H%M%SZ\n", time, utc) +
        absl::FormatTime("%Y%m%d/us-east-1/s3/aws4_request\n", time, utc) +
        "7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972";

    EXPECT_EQ(signing_string, expected_signing_string);

    std::string expected = "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41";
    auto signature = Signature(aws_secret_access_key, aws_region, time, expected_signing_string);
    EXPECT_EQ(signature, expected);
}

TEST(S3KeyValueStoreTest, AWSSignaturePutExample) {
    // https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
    // These values from worked exapmle in "Example: PUT Object" Section
    absl::CivilSecond cs(2013, 5, 24, 0, 0, 0);
    auto utc = absl::UTCTimeZone();
    auto time = absl::FromCivil(cs, utc);

    std::string aws_access_key = "AKIAIOSFODNN7EXAMPLE";
    std::string aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    std::string aws_region = "us-east-1";
    std::string bucket = "examplebucket";

    auto str_uri = absl::StrFormat("s3://%s/test$file.text", bucket);
    auto uri = ParseGenericUri(str_uri);
    auto host = absl::StrFormat("%s.s3.amazonaws.com", bucket);
    auto x_amz_date = absl::FormatTime("%Y%m%dT%H%M%SZ\n", time, utc);

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto canonical_request,
        CanonicalRequest(
            "PUT",
            uri,
            {
                { "Host", host },
                { "Date", "Fri, 24 May 2013 00:00:00 GMT" },
                { "x-amz-content-sha256", "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072" },
                { "x-amz-date", x_amz_date },
                { "x-amz-storage-class", "REDUCED_REDUNDANCY"}
            },
            "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072")
    );

    std::string expected_canonical_request =
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

    auto signing_string = SigningString(canonical_request, time, aws_region);

    std::string expected_signing_string =
        "AWS4-HMAC-SHA256\n" +
        absl::FormatTime("%Y%m%dT%H%M%SZ\n", time, utc) +
        absl::FormatTime("%Y%m%d/us-east-1/s3/aws4_request\n", time, utc) +
        "9e0e90d9c76de8fa5b200d8c849cd5b8dc7a3be3951ddb7f6a76b4158342019d";

    EXPECT_EQ(signing_string, expected_signing_string);

    std::string expected = "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd";
    auto signature = Signature(aws_secret_access_key, aws_region, time, expected_signing_string);
    EXPECT_EQ(signature, expected);
}


TEST(S3KeyValueStoreTest, CanonicalRequest) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result,
        CanonicalRequest(
            "PUT",
            ParseGenericUri("s3://bucket/path/to/file.jpg?qux=baz&foo=bar"),
            {
                { "x-amz-content-sha256", "abcdef0123456789" },
                { "X-amz-date", "20230330" },
                { "x-amz-expires", "3600" },
                { "Host", "bucket.s3.us-east-1.amazonaws.com" },
            },
            "abcdef0123456789"
        )
    );

    std::string expected =
        "PUT\n"
        "/path/to/file.jpg\n"
        "foo=bar&qux=baz\n"
        "host:bucket.s3.us-east-1.amazonaws.com\n"
        "x-amz-content-sha256:abcdef0123456789\n"
        "x-amz-date:20230330\n"
        "x-amz-expires:3600\n"
        "\n"
        "host;x-amz-content-sha256;x-amz-date;x-amz-expires\n"
        "abcdef0123456789";

    EXPECT_EQ(result, expected);
}

}
