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

#include "tensorstore/internal/json/json.h"

#include <stddef.h>

#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_json {

namespace {

struct CordJsonInputAdapter {
  using char_type = char;

  explicit CordJsonInputAdapter(const absl::Cord& cord)
      : chunks(cord.Chunks()), it(chunks.begin()), end(chunks.end()) {
    if (it != end) {
      cur = *it;
    }
  }

  std::char_traits<char>::int_type get_character() {
    while (cur.empty()) {
      if (it == end || ++it == end) {
        return std::char_traits<char>::eof();
      }
      cur = *it;
    }
    char c = cur.front();
    cur.remove_prefix(1);
    return std::char_traits<char>::to_int_type(c);
  }

  absl::Cord::ChunkRange chunks;
  absl::Cord::ChunkIterator it;
  absl::Cord::ChunkIterator end;
  std::string_view cur;
};

}  // namespace

::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   std::string_view name) {
  if (auto it = j_obj->find(name); it != j_obj->end()) {
    auto node = j_obj->extract(it);
    return std::move(node.mapped());
  }
  return ::nlohmann::json(::nlohmann::json::value_t::discarded);
}
absl::Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Object includes extra members: %s",
      absl::StrJoin(j_obj, ",", [](std::string* out, const auto& p) {
        absl::StrAppend(out, QuoteString(p.first));
      })));
}

::nlohmann::json ParseJson(std::string_view str) {
  return ::nlohmann::json::parse(str, nullptr, false);
}

::nlohmann::json ParseJson(const absl::Cord& cord) {
  if (auto flat = cord.TryFlat()) {
    return ::nlohmann::json::parse(*flat, nullptr, false);
  }
  ::nlohmann::json value;
  ::nlohmann::detail::parser<::nlohmann::json, CordJsonInputAdapter>(
      CordJsonInputAdapter(cord), nullptr, /*allow_exceptions=*/false,
      /*ignore_comments=*/false)
      .parse(/*strict=*/true, value);
  return value;
}

absl::Status JsonParseArray(
    const ::nlohmann::json& j,
    absl::FunctionRef<absl::Status(ptrdiff_t size)> size_callback,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& value,
                                   ptrdiff_t index)>
        element_callback) {
  const auto* j_array = j.get_ptr<const ::nlohmann::json::array_t*>();
  if (!j_array) {
    return internal_json::ExpectedError(j, "array");
  }
  const ptrdiff_t size = j_array->size();
  TENSORSTORE_RETURN_IF_ERROR(size_callback(size));
  for (DimensionIndex i = 0; i < size; ++i) {
    auto status = element_callback(j[i], i);
    TENSORSTORE_RETURN_IF_ERROR(status).Format(
        "Error parsing value at position %d", i);
  }
  return absl::OkStatus();
}

absl::Status JsonValidateArrayLength(ptrdiff_t parsed_size,
                                     ptrdiff_t expected_size) {
  if (parsed_size != expected_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Array has length %d but should have length %d",
                        parsed_size, expected_size));
  }
  return absl::OkStatus();
}

}  // namespace internal_json
}  // namespace tensorstore
