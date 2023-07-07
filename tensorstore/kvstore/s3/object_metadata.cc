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

#include "tensorstore/kvstore/s3/object_metadata.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_storage_s3 {

namespace {

/// ifrom hashlib import md5; md5("".encode("utf-8")).hexdigest()
static constexpr char kEmptyETag[] = "\"d41d8cd98f00b204e9800998ecf8427e\"";

}

using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::internal_http::TryParseIntHeader;
using ::tensorstore::internal_http::TryParseBoolHeader;

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result) {
  result->size =
      TryParseIntHeader<uint64_t>(headers, "content-length").value_or(0);

  result->deleted = TryParseBoolHeader(headers, "x-amz-delete-marker").value_or(false);

  if(auto it = headers.find("x-amz-version-id"); it != headers.end()) {
    result->version_id = it->second;
  }

  if(auto it = headers.find("last-modified"); it != headers.end()) {
    std::string error;
    absl::ParseTime("%a, %-d %b %Y %H:%M:%S %Z", it->second, &result->last_modified, &error);
  }

  if(auto it = headers.find("etag"); it != headers.end()) {
    result->etag = it->second;
  }
}

Result<StorageGeneration> ComputeGenerationFromHeaders(
    const std::multimap<std::string, std::string>& headers) {

  std::string last_modified;
  std::string etag;

  if(auto it = headers.find("last-modified"); it != headers.end()) {
    last_modified = it->second;
  } else {
    return absl::NotFoundError("last-modified not found in response headers");
  }

  if(auto it = headers.find("etag"); it != headers.end()) {
    etag = it->second;
  } else {
    return absl::NotFoundError("etag not found in response headers");
  }

  return StorageGeneration::FromString(
    tensorstore::StrCat(etag, ";", last_modified));
}

std::pair<std::string, std::string>
ExtractETagAndLastModified(const StorageGeneration & generation) {
  auto gen = StorageGeneration::DecodeString(generation);
  std::vector<std::string_view> parts = absl::StrSplit(gen, ";");
  if(parts.size() == 2) {
    return {std::string(parts[0]), std::string(parts[1])};
  }
  return {kEmptyETag, ""};
}


}  // namespace internal_storage_s3
}  // namespace tensorstore
