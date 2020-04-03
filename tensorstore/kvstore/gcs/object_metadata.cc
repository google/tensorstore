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

#include "tensorstore/kvstore/gcs/object_metadata.h"

#include <map>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::internal_storage_gcs::ObjectMetadata;

namespace tensorstore {
namespace internal {

// TODO: Move to internal/json
template <>
absl::optional<absl::Time> JsonValueAs(const ::nlohmann::json& json,
                                       bool strict) {
  if (!json.is_string()) {
    return absl::nullopt;
  }
  absl::Time time;
  if (absl::ParseTime(absl::RFC3339_full, json.get_ref<std::string const&>(),
                      &time, nullptr)) {
    return time;
  }
  return absl::nullopt;
}

}  // namespace internal
namespace {

template <typename T>
void JsonSetIfExists(T& value, const ::nlohmann::json& j,
                     const char* field_name) {
  using tensorstore::internal::JsonValueAs;

  auto it = j.find(field_name);
  if (it != j.end()) {
    auto field = JsonValueAs<T>(it.value());
    if (field) {
      value = field.value();
    }
  }
}
}  // namespace

namespace internal_storage_gcs {

void SetObjectMetadata(const ::nlohmann::json& json, ObjectMetadata* result) {
  JsonSetIfExists(result->bucket, json, "bucket");
  JsonSetIfExists(result->cache_control, json, "cacheControl");
  JsonSetIfExists(result->content_disposition, json, "contentDisposition");
  JsonSetIfExists(result->content_encoding, json, "contentEncoding");
  JsonSetIfExists(result->content_language, json, "contentLanguage");
  JsonSetIfExists(result->content_type, json, "contentType");
  JsonSetIfExists(result->crc32c, json, "crc32c");
  JsonSetIfExists(result->etag, json, "etag");
  JsonSetIfExists(result->id, json, "id");
  JsonSetIfExists(result->kind, json, "kind");
  JsonSetIfExists(result->kms_key_name, json, "kmsKeyName");
  JsonSetIfExists(result->md5_hash, json, "md5Hash");
  JsonSetIfExists(result->media_link, json, "mediaLink");
  JsonSetIfExists(result->name, json, "name");
  JsonSetIfExists(result->self_link, json, "selfLink");
  JsonSetIfExists(result->storage_class, json, "storageClass");

  JsonSetIfExists(result->size, json, "size");
  JsonSetIfExists(result->component_count, json, "componentCount");
  JsonSetIfExists(result->generation, json, "generation");
  JsonSetIfExists(result->metageneration, json, "metageneration");

  JsonSetIfExists(result->temporary_hold, json, "temporaryHold");
  JsonSetIfExists(result->event_based_hold, json, "eventBasedHold");

  JsonSetIfExists(result->retention_expiration_time, json,
                  "retentionExpirationTime");

  JsonSetIfExists(result->time_created, json, "timeCreated");

  JsonSetIfExists(result->updated, json, "updated");

  JsonSetIfExists(result->time_deleted, json, "timeDeleted");

  JsonSetIfExists(result->time_storage_class_updated, json,
                  "timeStorageClassUpdated");

  // Ignore "owner": { "entity", "entityId" }
  // Ignore "acl": { ... }
  // Ignore "customerEncryption": { ... }
  // Ignore "metadata": { ...}
}

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result) {
  auto set_int64_value = [&](const char* header, int64_t& output) {
    auto it = headers.find(header);
    if (it != headers.end()) {
      int64_t v = 0;
      if (absl::SimpleAtoi(it->second, &v)) {
        output = v;
      }
    }
  };

  auto set_uint64_value = [&](const char* header, uint64_t& output) {
    auto it = headers.find(header);
    if (it != headers.end()) {
      uint64_t v = 0;
      if (absl::SimpleAtoi(it->second, &v)) {
        output = v;
      }
    }
  };

  auto set_string_value = [&](const char* header, std::string& output) {
    auto it = headers.find(header);
    if (it != headers.end()) {
      output = it->second;
    }
  };

  set_uint64_value("content-length", result->size);
  set_int64_value("x-goog-generation", result->generation);
  set_int64_value("x-goog-metageneration", result->metageneration);

  set_string_value("content-type", result->content_type);
  set_string_value("x-goog-storage-class", result->storage_class);

  // goog hash is encoded as a list of k=v,k=v pairs.
  auto it = headers.find("x-goog-hash");
  if (it != headers.end()) {
    for (absl::string_view kv : absl::StrSplit(it->second, absl::ByChar(','))) {
      std::pair<absl::string_view, absl::string_view> split =
          absl::StrSplit(kv, absl::MaxSplits('=', 1));

      if (split.first == "crc32c") {
        result->crc32c = std::string(split.second);
      } else if (split.first == "md5") {
        result->md5_hash = std::string(split.second);
      }
    }
  }
}

Result<ObjectMetadata> ParseObjectMetadata(absl::string_view source) {
  auto json = internal::ParseJson(source);
  if (json.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse object metadata: ", source));
  }

  ObjectMetadata result{};
  // Initialize time to InfinitePast
  result.retention_expiration_time = result.time_created = result.updated =
      result.time_deleted = result.time_storage_class_updated =
          absl::InfinitePast();

  SetObjectMetadata(json, &result);
  return std::move(result);
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
