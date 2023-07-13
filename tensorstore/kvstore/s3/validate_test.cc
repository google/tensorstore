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

using ::tensorstore::internal_kvstore_s3::IsValidBucketName;
using ::tensorstore::internal_kvstore_s3::ClassifyBucketName;
using ::tensorstore::internal_kvstore_s3::BucketNameType;
using ::tensorstore::internal_kvstore_s3::IsValidObjectName;


TEST(ValidateTest, ClassifyBucketName) {
  // Standard bucket names are < 63 characters
  EXPECT_EQ(ClassifyBucketName("1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "123"), BucketNameType::Standard);

  // No underscores in standard bucket names
  EXPECT_EQ(ClassifyBucketName("1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1_3"), BucketNameType::Invalid);

  // No uppercase in standard bucket names
  EXPECT_EQ(ClassifyBucketName("1234567890ABCDEFGHIJ"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "123"), BucketNameType::Invalid);

  // Old us-east-1 style buckets are > 64 characters.
  // uppercase and underscores allowed
  EXPECT_EQ(ClassifyBucketName("1234567890ABCDEFGHIJ"
                               "1234567890abcdefghij"
                               "1234567890abcdefghij"
                               "1_34"), BucketNameType::OldUSEast1);
}

TEST(ValidateTest, IsValidBucketName) {
  EXPECT_TRUE(IsValidBucketName("foo"));
  EXPECT_TRUE(IsValidBucketName("a.b"));
  EXPECT_TRUE(IsValidBucketName("a-b"));

  // No IP Address style names
  EXPECT_FALSE(IsValidBucketName("192.168.0.1"));

  // Invalid prefixes and suffixes
  EXPECT_FALSE(IsValidBucketName("xn--foobar"));
  EXPECT_FALSE(IsValidBucketName("foobar-s3alias"));
  EXPECT_FALSE(IsValidBucketName("foobar--ol-s3"));

  // Uppercase and underscores allowed for names > 63
  // (old style us-east-1 buckets)
  EXPECT_TRUE(
      IsValidBucketName("1234567890b123456789012345678901234567890"
                        "1234567890b123456789012345678901234567890"
                        "ABCD_EFGH"));


  // But forbidden for names <= 63
  EXPECT_FALSE(IsValidBucketName("ABCD_EFGH"));

  EXPECT_FALSE(IsValidBucketName("."));     // <3
  EXPECT_FALSE(IsValidBucketName(".."));    // <3
  EXPECT_FALSE(IsValidBucketName("aa"));    // < 3
  EXPECT_FALSE(IsValidBucketName("-foo"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("foo-"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("a..b"));  // consecutive dots
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
