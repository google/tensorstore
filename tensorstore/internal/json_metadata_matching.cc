// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/json_metadata_matching.h"

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/json.h"

namespace tensorstore {
namespace internal {

absl::Status ValidateMetadataSubset(const ::nlohmann::json::object_t& expected,
                                    const ::nlohmann::json::object_t& actual) {
  for (const auto& [key, value] : expected) {
    auto it = actual.find(key);
    if (it == actual.end()) {
      return MetadataMismatchError(
          key, value, ::nlohmann::json(::nlohmann::json::value_t::discarded));
    }
    if (!internal_json::JsonSame(it->second, value)) {
      return MetadataMismatchError(key, value, it->second);
    }
  }
  return absl::OkStatus();
}
}  // namespace internal
}  // namespace tensorstore
