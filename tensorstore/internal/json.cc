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

#include "tensorstore/internal/json.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

Status MaybeAnnotateMemberError(const Status& status,
                                std::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(
      status, StrCat("Error parsing object member ", QuoteString(member_name)));
}

Status MaybeAnnotateMemberConvertError(const Status& status,
                                       std::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(status, StrCat("Error converting object member ",
                                            QuoteString(member_name)));
}

Status MaybeAnnotateArrayElementError(const Status& status, std::size_t i,
                                      bool is_loading) {
  return MaybeAnnotateStatus(
      status,
      tensorstore::StrCat("Error ", is_loading ? "parsing" : "converting",
                          " value at position ", i));
}

}  // namespace internal_json
namespace internal {

::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   std::string_view name) {
  if (auto it = j_obj->find(name); it != j_obj->end()) {
    auto node = j_obj->extract(it);
    return std::move(node.mapped());
  }
  return ::nlohmann::json(::nlohmann::json::value_t::discarded);
}
Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj) {
  return absl::InvalidArgumentError(
      StrCat("Object includes extra members: ",
             absl::StrJoin(j_obj, ",", [](std::string* out, const auto& p) {
               *out += QuoteString(p.first);
             })));
}

::nlohmann::json ParseJson(std::string_view str) {
  return ::nlohmann::json::parse(str, nullptr, false);
}

Status JsonParseArray(
    const ::nlohmann::json& j,
    absl::FunctionRef<Status(std::ptrdiff_t size)> size_callback,
    absl::FunctionRef<Status(const ::nlohmann::json& value,
                             std::ptrdiff_t index)>
        element_callback) {
  const auto* j_array = j.get_ptr<const ::nlohmann::json::array_t*>();
  if (!j_array) {
    return internal_json::ExpectedError(j, "array");
  }
  const std::ptrdiff_t size = j_array->size();
  TENSORSTORE_RETURN_IF_ERROR(size_callback(size));
  for (DimensionIndex i = 0; i < size; ++i) {
    auto status = element_callback(j[i], i);
    if (!status.ok()) {
      return MaybeAnnotateStatus(status,
                                 StrCat("Error parsing value at position ", i));
    }
  }
  return absl::OkStatus();
}

Status JsonValidateArrayLength(std::ptrdiff_t parsed_size,
                               std::ptrdiff_t expected_size) {
  if (parsed_size != expected_size) {
    return absl::InvalidArgumentError(StrCat("Array has length ", parsed_size,
                                             " but should have length ",
                                             expected_size));
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
