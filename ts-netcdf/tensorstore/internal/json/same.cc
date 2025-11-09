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

#include "tensorstore/internal/json/same.h"

#include <variant>

#include "absl/container/inlined_vector.h"
#include <nlohmann/json.hpp>

namespace tensorstore {
namespace internal_json {

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
