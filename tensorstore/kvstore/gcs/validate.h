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

#ifndef TENSORSTORE_KVSTORE_GCS_VALIDATE_H_
#define TENSORSTORE_KVSTORE_GCS_VALIDATE_H_

#include <string_view>

#include "tensorstore/kvstore/generation.h"

namespace tensorstore {
namespace internal_storage_gcs {

// Returns whether the bucket name is valid.
// https://cloud.google.com/storage/docs/naming#requirements
bool IsValidBucketName(std::string_view bucket);

// Returns whether the object name is a valid GCS object name.
bool IsValidObjectName(std::string_view name);

// Returns whether the StorageGeneration is valid for GCS.
bool IsValidStorageGeneration(const StorageGeneration& gen);

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_VALIDATE_H_
