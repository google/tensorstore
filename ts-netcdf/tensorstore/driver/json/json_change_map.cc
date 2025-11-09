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

#include "tensorstore/driver/json/json_change_map.h"

#include <string>
#include <string_view>
#include <utility>

#include "absl/container/btree_map.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_pointer.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_json_driver {

Result<::nlohmann::json> JsonChangeMap::Apply(
    const ::nlohmann::json& existing,
    std::string_view sub_value_pointer) const {
  Map::const_iterator changes_it = map_.lower_bound(sub_value_pointer),
                      changes_end = map_.end();
  // Check if `sub_value_pointer` is exactly present as a map entry.
  if (changes_it != changes_end && changes_it->first == sub_value_pointer) {
    // Exact match, map preconditions guarantee that there are no prefix
    // matches.

    // Validate preconditions on `existing`.
    TENSORSTORE_RETURN_IF_ERROR(
        json_pointer::Dereference(existing, sub_value_pointer,
                                  json_pointer::kSimulateCreate),
        internal::ConvertInvalidArgumentToFailedPrecondition(_));
    // avoid basic_json::operator ValueType()
    return {std::in_place, changes_it->second};
  }

  // `sub_value_pointer` is not exactly present in the map.

  if (changes_it != map_.begin()) {
    // Check if `sub_value_pointer` is contained within an entry in the map.
    auto prev_it = std::prev(changes_it);
    if (json_pointer::Compare(prev_it->first, sub_value_pointer) ==
        json_pointer::kContains) {
      // Determine the modified value.
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto* modified_value,
          json_pointer::Dereference(
              prev_it->second, sub_value_pointer.substr(prev_it->first.size()),
              json_pointer::kMustExist));
      // Validate preconditions of entry referred to by `prev_it`.
      TENSORSTORE_RETURN_IF_ERROR(
          json_pointer::Dereference(existing, prev_it->first,
                                    json_pointer::kSimulateCreate),
          internal::ConvertInvalidArgumentToFailedPrecondition(_));
      return {std::in_place, *modified_value};
    }
  }

  // `sub_value_pointer` is not contained with an entry in the map.

  // The new value is equal to the value in `existing` corresponding to
  // `sub_value_pointer`, if any, modified by any descendant entries of
  // `sub_value_pointer` that are present in the map.

  ::nlohmann::json new_value;
  {
    // Find value in `existing` corresponding to `sub_value_pointer`.
    TENSORSTORE_ASSIGN_OR_RETURN(
        const ::nlohmann::json* restricted_existing,
        json_pointer::Dereference(existing, sub_value_pointer,
                                  json_pointer::kSimulateCreate));
    if (restricted_existing) {
      new_value = *restricted_existing;
    } else {
      new_value = ::nlohmann::json(::nlohmann::json::value_t::discarded);
    }
  }

  // Iterate over all descendants of `sub_value_pointer` that are present in the
  // map and apply them to `new_value`.
  for (; changes_it != changes_end &&
         json_pointer::Compare(changes_it->first, sub_value_pointer) ==
             json_pointer::kContainedIn;
       ++changes_it) {
    TENSORSTORE_RETURN_IF_ERROR(
        json_pointer::Replace(new_value,
                              std::string_view(changes_it->first)
                                  .substr(sub_value_pointer.size()),
                              changes_it->second),
        internal::ConvertInvalidArgumentToFailedPrecondition(_));
  }
  return new_value;
}

bool JsonChangeMap::CanApplyUnconditionally(
    std::string_view sub_value_pointer) const {
  Map::const_iterator changes_it;
  if (sub_value_pointer.empty()) {
    changes_it = map_.begin();
  } else {
    changes_it = map_.lower_bound(sub_value_pointer);
  }
  if (changes_it != map_.end()) {
    // Check for exact match.
    if (changes_it->first == sub_value_pointer) {
      return true;
    }
  }
  if (changes_it != map_.begin()) {
    // Check for containment.
    auto prev_it = std::prev(changes_it);
    return json_pointer::Compare(prev_it->first, sub_value_pointer) ==
           json_pointer::kContains;
  }
  return false;
}

absl::Status JsonChangeMap::AddChange(std::string_view sub_value_pointer,
                                      ::nlohmann::json sub_value) {
  auto it = map_.lower_bound(sub_value_pointer);
  if (it != map_.end()) {
    // Check if this change contains another change.
    auto compare_result = json_pointer::Compare(sub_value_pointer, it->first);
    assert(compare_result <= json_pointer::kEqual);
    if (compare_result == json_pointer::kEqual) {
      // Json pointer is already present in `changes`, just replace the
      // associated value.
      it->second = std::move(sub_value);
      return absl::OkStatus();
    }
    // Delete any changes that are contained in `sub_value_pointer`, since they
    // will be overwritten.
    while (compare_result == json_pointer::kContains) {
      it = map_.erase(it);
      if (it == map_.end()) break;
      compare_result = json_pointer::Compare(sub_value_pointer, it->first);
    }
  }
  if (it != map_.begin()) {
    auto prev_it = std::prev(it);
    if (json_pointer::Compare(prev_it->first, sub_value_pointer) ==
        json_pointer::kContains) {
      // Key associated with `prev_it` contains `sub_value_pointer`.  Just apply
      // `sub_value` to `prev_it->second`.
      return json_pointer::Replace(
          prev_it->second, sub_value_pointer.substr(prev_it->first.size()),
          std::move(sub_value));
    }
  }
  map_.try_emplace(it, std::string(sub_value_pointer), std::move(sub_value));
  return absl::OkStatus();
}

}  // namespace internal_json_driver
}  // namespace tensorstore
