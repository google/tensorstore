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

#ifndef TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_
#define TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_

/// \file
/// Key-value store where each key corresponds to a S3 object and the value is
/// stored as the file content.

#include <stdint.h>

#include <map>
#include <string>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_s3 {

/// Partial metadata for a S3 object
/// https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingMetadata.html
struct ObjectMetadata {
  std::string md5;
  std::string etag;

  uint64_t size = 0;
  std::string version_id;
  bool deleted;

  absl::Time last_modified = absl::InfinitePast();
};

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result);


Result<StorageGeneration> ComputeGenerationFromHeaders(
    const std::multimap<std::string, std::string>& headers);

std::pair<std::string, std::string>
ExtractETagAndLastModified(const StorageGeneration & generation);

}  // namespace internal_storage_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_
