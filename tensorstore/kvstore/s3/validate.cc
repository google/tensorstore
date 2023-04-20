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

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"

#include "re2/re2.h"

#include "tensorstore/internal/ascii_utils.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/kvstore/s3/validate.h"

using ::tensorstore::internal::AsciiSet;

namespace tensorstore {
namespace internal_storage_s3 {

RE2 ip_address_re("^\\d+\\.\\d+\\.\\d+\\.\\d+$");

constexpr AsciiSet kS3BucketValidChars{
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    ".-"};

constexpr AsciiSet kOldS3BucketValidChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".-_"};

constexpr AsciiSet kS3ObjectSafeChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!-_.*'()"};

constexpr AsciiSet kS3ObjectSpecialChars{
    "&$@=;/:+ ,?"
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    "\x7f"};

// Returns whether the bucket name is valid.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
BucketNameType ClassifyBucketName(std::string_view bucket) {
  if (bucket.size() < 3 || bucket.size() > 255) return BucketNameType::Invalid;
  bool old_us_east = bucket.size() > 63;

  // Bucket names must start and end with a number or letter.
  if (!absl::ascii_isdigit(*bucket.begin()) &&
    (old_us_east ?
      !absl::ascii_isalpha(*bucket.begin()) :
      !absl::ascii_islower(*bucket.begin()))) {
    return BucketNameType::Invalid;
  }

  // Bucket names must start and end with a number or letter.
  if (!absl::ascii_isdigit(*bucket.rbegin()) &&
    (old_us_east ?
      !absl::ascii_isalpha(*bucket.rbegin()) :
      !absl::ascii_islower(*bucket.rbegin()))) {
    return BucketNameType::Invalid;
  }

  // No IP Address styles
  if(RE2::FullMatch(bucket, ip_address_re)) return BucketNameType::Invalid;
  if(absl::StartsWith(bucket, "xn--")) return BucketNameType::Invalid;
  // reserved for access point alias names
  if(absl::EndsWith(bucket, "-s3alias")) return BucketNameType::Invalid;
  // reserved for Object Lambda Access
  if(absl::EndsWith(bucket, "--ol-s3")) return BucketNameType::Invalid;

  std::string_view::value_type last_char = '\0';

  for (const auto ch: bucket) {
    if(old_us_east ? !kOldS3BucketValidChars.Test(ch) : !kS3BucketValidChars.Test(ch) ||
       (ch == '.' && last_char == '.')) {
         return BucketNameType::Invalid;
       }

    last_char = ch;
  }

  return old_us_east ? BucketNameType::OldUSEast1 : BucketNameType::Standard;
}

bool IsValidBucketName(std::string_view bucket) {
  return ClassifyBucketName(bucket) != BucketNameType::Invalid;
}

// Returns whether the object name is a valid S3 object name.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
bool IsValidObjectName(std::string_view name) {
  if (name.empty() || name.size() > 1024) return false;
  return internal::IsValidUtf8(name);
}

}  // namespace internal_storage_s3
}  // namespace tensorstore
