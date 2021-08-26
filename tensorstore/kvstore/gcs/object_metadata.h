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

#ifndef TENSORSTORE_KVSTORE_GCS_OBJECT_METADATA_H_
#define TENSORSTORE_KVSTORE_GCS_OBJECT_METADATA_H_

/// \file
/// Key-value store where each key corresponds to a GCS object and the value is
/// stored as the file content.

#include <stdint.h>

#include <map>
#include <string>
#include <string_view>

#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// Partial metadata for a GCS object
/// https://cloud.google.com/kvstore/docs/json_api/v1/objects#resource
struct ObjectMetadata {
  std::string name;
  std::string md5_hash;
  std::string crc32c;

  uint64_t size = 0;
  int64_t generation = 0;
  int64_t metageneration = 0;

  // RFC3339 format.
  absl::Time time_created = absl::InfinitePast();
  absl::Time updated = absl::InfinitePast();
  absl::Time time_deleted = absl::InfinitePast();

  // Additional metadata.
  // string: bucket, cache_control, content_disposition, content_encoding,
  //   content_language, content_type,  etag,  id,  kind,  kms_key_name,
  //   media_link,  self_link,  storage_class, crc32,
  // object: owner, acl, customer_encryption, metadata.
  // boolean: event_based_hold, temporary_hold,
  // time: retention_expiration_time, time_storage_class_updated.

  using ToJsonOptions = IncludeDefaults;
  using FromJsonOptions = internal_json_binding::NoOptions;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(
      ObjectMetadata, internal_storage_gcs::ObjectMetadata::FromJsonOptions,
      internal_storage_gcs::ObjectMetadata::ToJsonOptions)
};

Result<ObjectMetadata> ParseObjectMetadata(std::string_view source);

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result);

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_OBJECT_METADATA_H_
