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

#include "absl/strings/str_split.h"
#include "re2/re2.h"

#include "tensorstore/internal/utf8.h"

namespace tensorstore {
namespace internal_storage_s3 {

// Returns whether the bucket name is valid.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
BucketNameType ClassifyBucketName(std::string_view bucket) {
  if (bucket.size() < 3 || bucket.size() > 255) return BucketNameType::Invalid;
  bool old_us_east = bucket.size() > 63;

  static LazyRE2 kIpAddress = {"^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$"};
  static LazyRE2 kS3BucketValidChars = {"^[a-z0-9\\.-]+$"};         // Current
  static LazyRE2 kOldS3BucketValidChars = {"^[a-zA-Z0-9\\.-_]+$"};  // Old us-east-1 style

  // No IP Address styles
  if(RE2::FullMatch(bucket, *kIpAddress)) return BucketNameType::Invalid;
  if(absl::StartsWith(bucket, "xn--")) return BucketNameType::Invalid;
  // reserved for access point alias names
  if(absl::EndsWith(bucket, "-s3alias")) return BucketNameType::Invalid;
  // reserved for Object Lambda Access
  if(absl::EndsWith(bucket, "--ol-s3")) return BucketNameType::Invalid;

  // Bucket name is a series of labels split by .
  // https://web.archive.org/web/20170121163958/http://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
  for (std::string_view v : absl::StrSplit(bucket, absl::ByChar('.'))) {
    if (v.empty()) return BucketNameType::Invalid;
    // Bucket names must start and end with a number or letter.
    if (!absl::ascii_isdigit(*v.begin()) &&
        (old_us_east ? !absl::ascii_isalpha(*v.begin()) : !absl::ascii_islower(*v.begin()))) {
      return BucketNameType::Invalid;
    }
    if (!absl::ascii_isdigit(*v.rbegin()) &&
        (old_us_east ? !absl::ascii_isalpha(*v.rbegin()) : !absl::ascii_islower(*v.rbegin()))) {
      return BucketNameType::Invalid;
    }


    if(old_us_east ? !RE2::FullMatch(v, *kOldS3BucketValidChars) : !RE2::FullMatch(v, *kS3BucketValidChars)) {
      return BucketNameType::Invalid;
    }
  }

  return old_us_east ? BucketNameType::OldUSEast1 : BucketNameType::Standard;
}

// Returns whether the object name is a valid S3 object name.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
bool IsValidObjectName(std::string_view name) {
  if (name.empty() || name.size() > 1024) return false;
  return internal::IsValidUtf8(name);
}

// Returns whether the StorageGeneration is valid for s3 kvstore.
bool IsValidStorageGeneration(const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen) || StorageGeneration::IsNoValue(gen)) return true;
  auto etag = StorageGeneration::DecodeString(gen);
  if(absl::StartsWith(etag, "\"") && absl::EndsWith(etag, "\"")) return true;
  return false;
}


}  // namespace internal_storage_s3
}  // namespace tensorstore
