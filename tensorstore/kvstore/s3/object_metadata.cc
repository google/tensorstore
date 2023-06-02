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

  absl::Time last_modified;
  std::string etag;

  if(auto it = headers.find("last-modified"); it != headers.end()) {
    std::string error;
    if(!absl::ParseTime("%a, %d %b %Y %H:%M:%S %Z", it->second, &last_modified, &error)) {
      return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid last-modified: ", error));
    }
  } else {
    return absl::NotFoundError("last-modified not found in response headers");
  }

  if(auto it = headers.find("etag"); it != headers.end()) {
    etag = it->second;
  } else {
    return absl::NotFoundError("etag not found in response headers");
  }

  return StorageGeneration::FromString(
    tensorstore::StrCat(etag,
                        absl::FormatTime("%Y%m%dT%H%M%SZ", last_modified, absl::UTCTimeZone())));
}


}  // namespace internal_storage_s3
}  // namespace tensorstore
