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
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_storage_gcs {

using ::tensorstore::internal_json_binding::DefaultInitializedValue;

namespace jb = tensorstore::internal_json_binding;

inline constexpr auto ObjectMetadataBinder = jb::Object(
    jb::Member("name", jb::Projection(&ObjectMetadata::name)),
    jb::Member("md5Hash", jb::Projection(&ObjectMetadata::md5_hash,
                                         DefaultInitializedValue())),
    jb::Member("crc32c", jb::Projection(&ObjectMetadata::crc32c,
                                        DefaultInitializedValue())),
    jb::Member("size", jb::Projection(&ObjectMetadata::size,
                                      jb::DefaultInitializedValue(
                                          jb::LooseValueAsBinder))),
    jb::Member("generation", jb::Projection(&ObjectMetadata::generation,
                                            jb::DefaultInitializedValue(
                                                jb::LooseValueAsBinder))),
    jb::Member("metageneration", jb::Projection(&ObjectMetadata::metageneration,
                                                jb::DefaultInitializedValue(
                                                    jb::LooseValueAsBinder))),

    // RFC3339 format.
    jb::Member("timeCreated", jb::Projection(&ObjectMetadata::time_created,
                                             jb::DefaultValue([](auto* x) {
                                               *x = absl::InfinitePast();
                                             }))),
    jb::Member("updated", jb::Projection(&ObjectMetadata::updated,
                                         jb::DefaultValue([](auto* x) {
                                           *x = absl::InfinitePast();
                                         }))),
    jb::Member("timeDeleted", jb::Projection(&ObjectMetadata::time_deleted,
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

  set_uint64_value("content-length", result->size);
  set_int64_value("x-goog-generation", result->generation);
  set_int64_value("x-goog-metageneration", result->metageneration);

  // Ignore: content-type, x-goog-storage-class

  // goog hash is encoded as a list of k=v,k=v pairs.
  auto it = headers.find("x-goog-hash");
  if (it != headers.end()) {
    for (std::string_view kv : absl::StrSplit(it->second, absl::ByChar(','))) {
      std::pair<std::string_view, std::string_view> split =
          absl::StrSplit(kv, absl::MaxSplits('=', 1));

      if (split.first == "crc32c") {
        result->crc32c = std::string(split.second);
      } else if (split.first == "md5") {
        result->md5_hash = std::string(split.second);
      }
    }
  }
}

Result<ObjectMetadata> ParseObjectMetadata(std::string_view source) {
  auto json = internal::ParseJson(source);
  if (json.is_discarded()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Failed to parse object metadata: ", source));
  }

  return jb::FromJson<ObjectMetadata>(std::move(json));
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
