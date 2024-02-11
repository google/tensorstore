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

#include "tensorstore/kvstore/s3/validate.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_kvstore_s3::BucketNameType;
using ::tensorstore::internal_kvstore_s3::ClassifyBucketName;
using ::tensorstore::internal_kvstore_s3::IsValidBucketName;
using ::tensorstore::internal_kvstore_s3::IsValidObjectName;

TEST(ValidateTest, ClassifyBucketName) {
  EXPECT_EQ(ClassifyBucketName("aa"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("a..b"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("xn--bucket"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("bucket-s3alias"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("bucket--ol-s3"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("sthree-bucket"), BucketNameType::kInvalid);
  EXPECT_EQ(ClassifyBucketName("sthree-configurator.bucket"),
            BucketNameType::kInvalid);

  // Standard bucket names are < 63 characters
  EXPECT_EQ(ClassifyBucketName("1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "123"),
            BucketNameType::kStandard);

  // Old us-east-1 style buckets allow uppercase and underscores.
  EXPECT_EQ(ClassifyBucketName("1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1_3"),
            BucketNameType::kOldUSEast1);

  // No uppercase in standard bucket names
  EXPECT_EQ(ClassifyBucketName("1234567890ABCDEFGHIJ"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "123"),
            BucketNameType::kOldUSEast1);

  EXPECT_EQ(ClassifyBucketName("1234567890ABCDEFGHIJ"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1_34"),
            BucketNameType::kOldUSEast1);
}

TEST(ValidateTest, IsValidBucketName) {
  EXPECT_TRUE(IsValidBucketName("foo"));
  EXPECT_TRUE(IsValidBucketName("a.b"));
  EXPECT_TRUE(IsValidBucketName("a-1"));
  EXPECT_TRUE(IsValidBucketName("fo1.a-b"));

  // No IP Address style names
  EXPECT_FALSE(IsValidBucketName("192.168.5.4"));

  // Invalid prefixes and suffixes
  EXPECT_FALSE(IsValidBucketName("xn--bucket"));
  EXPECT_FALSE(IsValidBucketName("bucket-s3alias"));
  EXPECT_FALSE(IsValidBucketName("bucket--ol-s3"));
  EXPECT_FALSE(IsValidBucketName("sthree-bucket"));
  EXPECT_FALSE(IsValidBucketName("sthree-configurator.bucket"));

  EXPECT_FALSE(IsValidBucketName("foo$bar"));
  EXPECT_FALSE(IsValidBucketName("."));     // <3
  EXPECT_FALSE(IsValidBucketName(".."));    // <3
  EXPECT_FALSE(IsValidBucketName("aa"));    // <3
  EXPECT_FALSE(IsValidBucketName("-foo"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("foo-"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("a..b"));  // consecutive dots
  EXPECT_FALSE(IsValidBucketName("foo..bar"));
  EXPECT_FALSE(IsValidBucketName("foo.-bar"));

  // if a bucket has uppercase characters, then it is an
  // old style US-East-1 bucket created before 2018.
  EXPECT_TRUE(
      IsValidBucketName("1234567890b123456789012345678901234567890"
                        "1234567890b123456789012345678901234567890"
                        "ABCD_EFGH"));

  EXPECT_TRUE(IsValidBucketName("ABCD_EFGH"));
  EXPECT_TRUE(IsValidBucketName("abcd_efgh"));
}

TEST(ValidateTest, IsValidObjectName) {
  EXPECT_TRUE(IsValidObjectName("foo"));
  EXPECT_TRUE(IsValidObjectName("foo.bar"));

  EXPECT_FALSE(IsValidObjectName(""));
  EXPECT_TRUE(IsValidObjectName("."));
  EXPECT_TRUE(IsValidObjectName(".."));

  EXPECT_TRUE(IsValidObjectName("foo\rbar"));
  EXPECT_TRUE(IsValidObjectName("foo\nbar"));

  // Not utf-8
  EXPECT_FALSE(IsValidObjectName("\xfe\xfe\xff\xff"));
  EXPECT_FALSE(IsValidObjectName("\xfc\x80\x80\x80\x80\xaf"));
}

}  // namespace
