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

#include "tensorstore/internal/utf8.h"
#include "tensorstore/kvstore/s3/validate.h"

namespace tensorstore {
namespace internal_storage_s3 {

RE2 ip_address_re("^\\d+\\.\\d+\\.\\d+\\.\\d+$");


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

  unsigned char last_char = '\0';

  for (const auto ch : bucket) {
    // Bucket names can consist only of lowercase letters, numbers, dots (.), and hyphens (-).
    // except for old us-east-1 bucket names which can contain uppercase characters and underscores
    if (ch != '.' && ch != '-' && !absl::ascii_isdigit(ch) &&
        (old_us_east ? ch != '_' && !absl::ascii_isalpha(ch) : !absl::ascii_islower(ch)))
        return BucketNameType::Invalid;

    if(ch == '.' and last_char == '.') {
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
  return true;
}

}  // namespace internal_storage_s3
}  // namespace tensorstore
