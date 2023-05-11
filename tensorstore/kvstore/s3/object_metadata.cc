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
#include <nlohmann/json.hpp>
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_storage_s3 {

using ::tensorstore::internal_http::TryParseIntHeader;
using ::tensorstore::internal_http::TryParseBoolHeader;
using ::tensorstore::internal_json_binding::DefaultInitializedValue;

namespace jb = tensorstore::internal_json_binding;

inline constexpr auto ObjectMetadataBinder = jb::Object(
    jb::Member("x-amz-checksum-crc32c", jb::Projection(&ObjectMetadata::crc32c,
                                        DefaultInitializedValue())),
    jb::Member("x-amz-version-id", jb::Projection(&ObjectMetadata::version_id,
                                         DefaultInitializedValue())),
    jb::Member("x-amz-delete-marker", jb::Projection(&ObjectMetadata::deleted,
                                         DefaultInitializedValue())),
    jb::Member("Content-Length", jb::Projection(&ObjectMetadata::size,
                                      jb::DefaultInitializedValue(
                                          jb::LooseValueAsBinder))),

    jb::Member("Last-Modified", jb::Projection(&ObjectMetadata::time_modified,
                                             jb::DefaultValue([](auto* x) {
                                               *x = absl::InfinitePast();
                                             }))),
    jb::DiscardExtraMembers);

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ObjectMetadata,
                                       [](auto is_loading, const auto& options,
                                          auto* obj, ::nlohmann::json* j) {
                                         return ObjectMetadataBinder(
                                             is_loading, options, obj, j);
                                       })

void SetObjectMetadataFromHeaders(
    const std::multimap<std::string, std::string>& headers,
    ObjectMetadata* result) {
  result->size =
      TryParseIntHeader<uint64_t>(headers, "content-length").value_or(0);

  auto version_id_it = headers.find("x-amz-version-id");

  if (version_id_it != headers.end()) {
    result->version_id = version_id_it->second;
  }

  auto crc32c_it = headers.find("x-amz-checksum-crc32");

  if(crc32c_it != headers.end()) {
    result->crc32c = crc32c_it->second;
  }

  result->deleted = TryParseBoolHeader(headers, "x-amz-delete-marker").value_or(false);
}

Result<ObjectMetadata> ParseObjectMetadata(std::string_view source) {
  auto json = internal::ParseJson(source);
  if (json.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Failed to parse object metadata: ", source));
  }

  return jb::FromJson<ObjectMetadata>(std::move(json));
}

}  // namespace internal_storage_s3
}  // namespace tensorstore
