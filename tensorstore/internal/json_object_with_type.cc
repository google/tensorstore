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

#include "tensorstore/internal/json_object_with_type.h"

#include <ostream>
#include <string>
#include <utility>

#include "absl/base/macros.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/to_string.h"

namespace tensorstore {
namespace internal {

namespace {
constexpr const char* kTypeKey = "$type";

}  // namespace

JsonObjectWithType::JsonObjectWithType(std::string type,
                                       ::nlohmann::json::object_t object)
    : type(std::move(type)), object(std::move(object)) {
  ABSL_ASSERT(this->object.count(kTypeKey) == 0);
}

bool operator==(const JsonObjectWithType& a, const JsonObjectWithType& b) {
  return a.type == b.type && a.object == b.object;
}

std::ostream& operator<<(std::ostream& os, const JsonObjectWithType& x) {
  return os << ::nlohmann::json(x).dump();
}

Result<JsonObjectWithType> JsonObjectWithType::Parse(::nlohmann::json json) {
  auto* obj = json.get_ptr<::nlohmann::json::object_t*>();
  if (obj) {
    auto it = obj->find(kTypeKey);
    if (it != obj->end() && it->second.is_string()) {
      std::string type = std::move(*it->second.get_ptr<std::string*>());
      obj->erase(it);
      return JsonObjectWithType(std::move(type), std::move(*obj));
    }
  }
  return absl::InvalidArgumentError(
      StrCat("Expected object with string ", QuoteString(kTypeKey),
             " member, but received: ", json.dump()));
}

void to_json(::nlohmann::json& out,  // NOLINT
             JsonObjectWithType x) {
  x.object.emplace(kTypeKey, std::move(x.type));
  out = std::move(x.object);
}

}  // namespace internal
}  // namespace tensorstore
