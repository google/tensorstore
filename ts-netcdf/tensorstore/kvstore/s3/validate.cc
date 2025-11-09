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

#include <string_view>

#include "absl/strings/match.h"
#include "re2/re2.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/kvstore/generation.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

// Returns whether the bucket name is valid.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
BucketNameType ClassifyBucketName(std::string_view bucket) {
  // Bucket name is a series of labels split by .
  // https://web.archive.org/web/20170121163958/http://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
  // https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html

  if (bucket.size() < 3 || bucket.size() > 255 ||
      absl::EndsWith(bucket, "--ol-s3") ||   // reserved: Object Lambda Access
      absl::EndsWith(bucket, "-s3alias") ||  // reserved: access point aliases
      absl::StartsWith(bucket, "sthree-") ||
      absl::StartsWith(bucket, "sthree-configurator.") ||
      absl::StartsWith(bucket, "xn--")) {
    // bucket name invalid size or reserved.
    return BucketNameType::kInvalid;
  }

  static LazyRE2 kIpAddress = {"^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$"};

  // No IP Address style names.
  if (RE2::FullMatch(bucket, *kIpAddress)) {
    return BucketNameType::kInvalid;
  }

  static LazyRE2 kCurrentStyle = {
      "^([a-z0-9]([a-z0-9-]*[a-z0-9])?)"
      "([.]([a-z0-9]([a-z0-9-]*[a-z0-9])?))*$"};

  if (bucket.size() <= 63 && RE2::FullMatch(bucket, *kCurrentStyle)) {
    return BucketNameType::kStandard;
  }

  // Before March 1, 2018, buckets created in the US East (N. Virginia) Region
  // could have names that were up to 255 characters long and included uppercase
  // letters and underscores.
  static LazyRE2 kOldUSEastStyle = {
      "^([a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?)"
      "([.]([a-zA-Z0-9]([a-zA-Z0-9_-]*[a-zA-Z0-9])?))*$"};

  return RE2::FullMatch(bucket, *kOldUSEastStyle) ? BucketNameType::kOldUSEast1
                                                  : BucketNameType::kInvalid;
}

// Returns whether the object name is a valid S3 object name.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
// NOTE: Amazon recommends that object names do not include the following:
// ASCII characters 128-255 as well as any of: "'\{}^%`<>~[]#|
bool IsValidObjectName(std::string_view name) {
  if (name.empty() || name.size() > 1024) return false;
  return internal::IsValidUtf8(name);
}

// Returns whether the StorageGeneration is valid for s3 kvstore.
bool IsValidStorageGeneration(const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen) || StorageGeneration::IsNoValue(gen))
    return true;
  auto etag = StorageGeneration::DecodeString(gen);
  if (absl::StartsWith(etag, "\"") && absl::EndsWith(etag, "\"")) return true;
  return false;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
