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

#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#include "tensorstore/kvstore/s3/s3_metadata.h"
#include "tensorstore/util/status_testutil.h"


using ::tensorstore::internal_kvstore_s3::FindTag;
using ::tensorstore::internal_kvstore_s3::GetTag;
using ::tensorstore::Result;

namespace {

/// Exemplar ListObjects v2 Response
/// https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html#API_ListObjectsV2_ResponseSyntax
static constexpr char list_xml[] =
    R"(<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">)"
    R"(<Name>i-dont-exist</Name>)"
    R"(<Prefix>tensorstore/test/</Prefix>)"
    R"(<KeyCount>3</KeyCount>)"
    R"(<MaxKeys>1000</MaxKeys>)"
    R"(<IsTruncated>false</IsTruncated>)"
    R"(<Contents>)"
    R"(<Key>tensorstore/test/abc</Key>)"
    R"(<LastModified>2023-07-08T15:26:55.000Z</LastModified>)"
    R"(<ETag>&quot;900150983cd24fb0d6963f7d28e17f72&quot;</ETag>)"
    R"(<ChecksumAlgorithm>SHA256</ChecksumAlgorithm>)"
    R"(<Size>3</Size>)"
    R"(<StorageClass>STANDARD</StorageClass>)"
    R"(</Contents>)"
    R"(<Contents>)"
    R"(<Key>tensorstore/test/abcd</Key>)"
    R"(<LastModified>2023-07-08T15:26:55.000Z</LastModified>)"
    R"(<ETag>&quot;e2fc714c4727ee9395f324cd2e7f331f&quot;</ETag>)"
    R"(<ChecksumAlgorithm>SHA256</ChecksumAlgorithm>)"
    R"(<Size>4</Size>)"
    R"(<StorageClass>STANDARD</StorageClass>)"
    R"(</Contents>)"
    R"(<Contents>)"
    R"(<Key>tensorstore/test/abcde</Key>)"
    R"(<LastModified>2023-07-08T15:26:55.000Z</LastModified>)"
    R"(<ETag>&quot;ab56b4d92b40713acc5af89985d4b786&quot;</ETag>)"
    R"(<ChecksumAlgorithm>SHA256</ChecksumAlgorithm>)"
    R"(<Size>5</Size>)"
    R"(<StorageClass>STANDARD</StorageClass>)"
    R"(</Contents>)"
    R"(</ListBucketResult>)";

static constexpr char list_bucket_tag[] =
    R"(<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">)";

TEST(XmlSearchTest, TagSearch) {
    // We can find the initial tag
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto start_pos, FindTag(list_xml, list_bucket_tag));
    EXPECT_EQ(start_pos, 0);

    // Get the position after the initial tag
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(start_pos, FindTag(list_xml, list_bucket_tag, 0, false));
    EXPECT_EQ(start_pos, strlen(list_bucket_tag));

    // Searching for the initial tag at position 1 fails
    EXPECT_FALSE(FindTag(list_xml, list_bucket_tag, 1).ok());

    // We can find and parse the number of keys
    auto pos = start_pos;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto key_count_sv, GetTag(list_xml, "<KeyCount>", "</KeyCount>", &pos));
    auto key_count = std::stol(std::string(key_count_sv));
    EXPECT_EQ(key_count, 3);

    std::vector<std::string_view> keys;

    for (std::size_t i = 0; i < key_count; ++i) {
        // Find the next Contents section and Object Key within
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(pos, FindTag(list_xml, "<Contents>", pos, false));
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto key, GetTag(list_xml, "<Key>", "</Key>", &pos));
        keys.push_back(key);
    }

    EXPECT_THAT(keys, ::testing::ElementsAre(
        "tensorstore/test/abc",
        "tensorstore/test/abcd",
        "tensorstore/test/abcde"
    ));
}


}  // namespace
