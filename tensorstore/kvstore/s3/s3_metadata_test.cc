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

#include "tensorstore/kvstore/s3/s3_metadata.h"

#include <gtest/gtest.h>
#include "tinyxml2.h"

namespace {

using ::tensorstore::internal_kvstore_s3::GetNodeText;

/// Exemplar ListObjects v2 Response
/// https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html#API_ListObjectsV2_ResponseSyntax
static constexpr char kListXml[] =
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
    R"(<Key>tensorstore/test/ab&gt;cd</Key>)"
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

TEST(XmlSearchTest, GetNodeText) {
  tinyxml2::XMLDocument xmlDocument;
  ASSERT_EQ(xmlDocument.Parse(kListXml), tinyxml2::XML_SUCCESS);

  auto* root = xmlDocument.FirstChildElement("ListBucketResult");
  ASSERT_NE(root, nullptr);

  EXPECT_EQ("i-dont-exist", GetNodeText(root->FirstChildElement("Name")));
  auto* contents = root->FirstChildElement("Contents");
  ASSERT_NE(contents, nullptr);

  EXPECT_EQ(R"("900150983cd24fb0d6963f7d28e17f72")",
            GetNodeText(contents->FirstChildElement("ETag")));
}

}  // namespace
