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

#ifndef TENSORSTORE_KVSTORE_S3_VALIDATE_H_
#define TENSORSTORE_KVSTORE_S3_VALIDATE_H_

#include <string_view>

#include "tensorstore/kvstore/generation.h"

using ::tensorstore::StorageGeneration;

namespace tensorstore {
namespace internal_storage_s3 {

enum BucketNameType {
  Invalid = 0,
  Standard = 1,
  OldUSEast1 = 2,
};

// Distinguish between Invalid, Standard and Old us-east-1 buckets
BucketNameType ClassifyBucketName(std::string_view bucket);

// Returns whether the bucket name is valid.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
inline bool IsValidBucketName(std::string_view bucket) {
  return ClassifyBucketName(bucket) != BucketNameType::Invalid;
}

// Returns whether the object name is a valid S3 object name.
// https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
bool IsValidObjectName(std::string_view name);

// Returns whether the storage generation is valie
bool IsValidStorageGeneration(const StorageGeneration& gen);

}  // namespace internal_storage_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_VALIDATE_H_
