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

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// Metadata for a GCS object
/// https://cloud.google.com/kvstore/docs/json_api/v1/objects#resource
struct ObjectMetadata {
  std::string bucket;
  std::string cache_control;
  std::string content_disposition;
  std::string content_encoding;
  std::string content_language;
  std::string content_type;
  std::string crc32c;
  std::string etag;
  std::string id;
  std::string kind;
  std::string kms_key_name;
  std::string md5_hash;
  std::string media_link;
  std::string name;
  std::string self_link;
  std::string storage_class;

  uint64_t size = 0;
  int64_t component_count = 0;
  int64_t generation = 0;
  int64_t metageneration = 0;

  bool event_based_hold = false;
  bool temporary_hold = false;

  // RFC3339 format.
  absl::Time retention_expiration_time = absl::InfinitePast();
  absl::Time time_created = absl::InfinitePast();
  absl::Time updated = absl::InfinitePast();
  absl::Time time_deleted = absl::InfinitePast();
  absl::Time time_storage_class_updated = absl::InfinitePast();

  // Additional metadata.
  //   owner, acl, customer_encryption, metadata.
};

Result<ObjectMetadata> ParseObjectMetadata(absl::string_view source);

void SetObjectMetadata(const ::nlohmann::json& json, ObjectMetadata* result);

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result);

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_OBJECT_METADATA_H_
