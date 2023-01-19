// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs/validate.h"

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal_storage_gcs::IsValidBucketName;
using ::tensorstore::internal_storage_gcs::IsValidObjectName;

TEST(ValidateTest, IsValidBucketName) {
  EXPECT_TRUE(IsValidBucketName("foo"));
  EXPECT_TRUE(IsValidBucketName("a.b"));
  EXPECT_TRUE(IsValidBucketName("a-b"));
  EXPECT_TRUE(IsValidBucketName("1.2.3.4"));

  EXPECT_FALSE(IsValidBucketName("_abc"));
  EXPECT_FALSE(IsValidBucketName("abc_"));
  EXPECT_FALSE(
      IsValidBucketName("1234567890b123456789012345678901234567890"
                        "1234567890b123456789012345678901234567890"
                        "abcd"));  // part >63

  EXPECT_TRUE(IsValidBucketName("a._b"));
  EXPECT_TRUE(IsValidBucketName("a_.b"));

  EXPECT_FALSE(IsValidBucketName("."));     // <3
  EXPECT_FALSE(IsValidBucketName(".."));    // <3
  EXPECT_FALSE(IsValidBucketName("aa"));    // < 3
  EXPECT_FALSE(IsValidBucketName("_foo"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("foo_"));  // not number or letter.
  EXPECT_FALSE(IsValidBucketName("a..b"));  // empty dot
  EXPECT_FALSE(IsValidBucketName("a.-b"));  // part starts with -
  EXPECT_FALSE(IsValidBucketName("a-.b"));  // part ends with -
  EXPECT_FALSE(
      IsValidBucketName("1234567890b123456789012345678901234567890"
                        "1234567890b123456789012345678901234567890"
                        "abcd.b"));  // part >63
}

TEST(ValidateTest, IsValidObjectName) {
  EXPECT_TRUE(IsValidObjectName("foo"));
  EXPECT_TRUE(IsValidObjectName("foo.bar"));

  EXPECT_FALSE(IsValidObjectName(""));
  EXPECT_FALSE(IsValidObjectName("."));
  EXPECT_FALSE(IsValidObjectName(".."));
  EXPECT_FALSE(IsValidObjectName(".well-known/acme-challenge"));

  EXPECT_FALSE(IsValidObjectName("foo\rbar"));
  EXPECT_FALSE(IsValidObjectName("foo\nbar"));

  // Allowed, but discouraged.
  EXPECT_TRUE(IsValidObjectName("foo[*?#]"));

  // ascii iscontrol
  EXPECT_FALSE(IsValidObjectName("foo\004bar"));
  EXPECT_FALSE(IsValidObjectName("foo\tbar"));

  // Not utf-8
  EXPECT_FALSE(IsValidObjectName("\xfe\xfe\xff\xff"));
  EXPECT_FALSE(IsValidObjectName("\xfc\x80\x80\x80\x80\xaf"));
}

}  // namespace
