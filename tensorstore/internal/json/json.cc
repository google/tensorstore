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

#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json {

::nlohmann::json JsonExtractMember(::nlohmann::json::object_t* j_obj,
                                   std::string_view name) {
  if (auto it = j_obj->find(name); it != j_obj->end()) {
    auto node = j_obj->extract(it);
    return std::move(node.mapped());
  }
  return ::nlohmann::json(::nlohmann::json::value_t::discarded);
}
absl::Status JsonExtraMembersError(const ::nlohmann::json::object_t& j_obj) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Object includes extra members: ",
      absl::StrJoin(j_obj, ",", [](std::string* out, const auto& p) {
        *out += QuoteString(p.first);
      })));
}

::nlohmann::json ParseJson(std::string_view str) {
  return ::nlohmann::json::parse(str, nullptr, false);
}

absl::Status JsonParseArray(
    const ::nlohmann::json& j,
    absl::FunctionRef<absl::Status(std::ptrdiff_t size)> size_callback,
    absl::FunctionRef<absl::Status(const ::nlohmann::json& value,
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
      return MaybeAnnotateStatus(
          status, tensorstore::StrCat("Error parsing value at position ", i));
    }
  }
  return absl::OkStatus();
}

absl::Status JsonValidateArrayLength(std::ptrdiff_t parsed_size,
                                     std::ptrdiff_t expected_size) {
  if (parsed_size != expected_size) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Array has length ", parsed_size,
                            " but should have length ", expected_size));
  }
  return absl::OkStatus();
}

bool JsonSame(const ::nlohmann::json& a, const ::nlohmann::json& b) {
  using value_t = ::nlohmann::json::value_t;
  using array_t = ::nlohmann::json::array_t;
  using object_t = ::nlohmann::json::object_t;
  struct ArrayIterators {
    array_t::const_iterator a_cur, a_end, b_cur;
  };
  struct ObjectIterators {
    object_t::const_iterator a_cur, a_end, b_cur;
  };
  using StackEntry = std::variant<ArrayIterators, ObjectIterators>;
  absl::InlinedVector<StackEntry, 64> stack;
  const auto compare_or_defer_values = [&](const ::nlohmann::json& a_value,
                                           const ::nlohmann::json& b_value) {
    const auto t = a_value.type();
    switch (t) {
      case value_t::discarded:
      case value_t::null:
        return b_value.type() == t;
      case value_t::array: {
        if (b_value.type() != t) return false;
        const auto& a_arr = a_value.get_ref<const array_t&>();
        const auto& b_arr = b_value.get_ref<const array_t&>();
        if (a_arr.size() != b_arr.size()) return false;
        if (a_arr.empty()) return true;
        stack.emplace_back(
            ArrayIterators{a_arr.begin(), a_arr.end(), b_arr.begin()});
        return true;
      }
      case value_t::object: {
        if (b_value.type() != t) return false;
        const auto& a_obj = a_value.get_ref<const object_t&>();
        const auto& b_obj = b_value.get_ref<const object_t&>();
        if (a_obj.size() != b_obj.size()) return false;
        if (a_obj.empty()) return true;
        stack.emplace_back(
            ObjectIterators{a_obj.begin(), a_obj.end(), b_obj.begin()});
        return true;
      }
      default:
        return a_value == b_value;
    }
  };
  if (!compare_or_defer_values(a, b)) return false;
  while (!stack.empty()) {
    auto& e = stack.back();
    if (auto* array_iterators = std::get_if<ArrayIterators>(&e)) {
      auto& a_v = *array_iterators->a_cur;
      auto& b_v = *array_iterators->b_cur;
      if (++array_iterators->a_cur == array_iterators->a_end) {
        stack.pop_back();
      } else {
        ++array_iterators->b_cur;
      }
      if (!compare_or_defer_values(a_v, b_v)) {
        return false;
      }
    } else {
      auto* object_iterators = std::get_if<ObjectIterators>(&e);
      auto& a_kv = *object_iterators->a_cur;
      auto& b_kv = *object_iterators->b_cur;
      if (++object_iterators->a_cur == object_iterators->a_end) {
        stack.pop_back();
      } else {
        ++object_iterators->b_cur;
      }
      if (a_kv.first != b_kv.first ||
          !compare_or_defer_values(a_kv.second, b_kv.second)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace internal_json
}  // namespace tensorstore
