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

#ifndef TENSORSTORE_DRIVER_JSON_JSON_CHANGE_MAP_H_
#define TENSORSTORE_DRIVER_JSON_JSON_CHANGE_MAP_H_

#include <string>
#include <string_view>

#include "absl/container/btree_map.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_pointer.h"

namespace tensorstore {
namespace internal_json_driver {

/// Collection of changes to apply to a JSON value.
///
/// Each changes is specified as a pair of a JSON Pointer and a replacement JSON
/// value.  A replacement JSON value of `discarded` indicates a deletion.
class JsonChangeMap {
 private:
  struct MapCompare {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const {
      return json_pointer::Compare(a, b) < json_pointer::kEqual;
    }
  };

 public:
  /// Representation of changes as an interval map.  The key is a JSON Pointer,
  /// the value is the replacement value.
  ///
  /// Keys are ordered according to JSON Pointer ordering (i.e. lexicographical
  /// on the decoded sequence of reference tokens).
  ///
  /// The map is kept normalized: it is not allowed to contain two keys, where
  /// one contains the other.  For example, if a change is added for "/a/b" and
  /// "/a/c", and then for "/a", the "/a/b" and "/a/c" changes are removed from
  /// the map since they would be overwritten by the later value for "/a".
  /// Similarly, if there is a change for "/a" and later a change for "/a/b" is
  /// added, the change for "/a/b" is applied to the value stored for "/a", and
  /// no separate entry for "/a/b" is added.
  using Map = absl::btree_map<std::string, ::nlohmann::json, MapCompare>;

  /// Returns the portion specified by `sub_value_pointer` of the result of
  /// applying this change map to `existing`.
  ///
  /// \param existing Existing JSON value.
  /// \param sub_value_pointer JSON Pointer specifying the portion of the
  ///     result to return.
  /// \returns The value corresponding to `sub_value_pointer` with changes
  ///     applied.
  /// \error `absl::StatusCode::kNotFound` if `sub_value_pointer` is not
  ///     found.
  /// \error `absl::StatusCode::kFailedPrecondition` if `existing` is not
  ///     compatible with some of the changes.
  Result<::nlohmann::json> Apply(const ::nlohmann::json& existing,
                                 std::string_view sub_value_pointer = {}) const;

  /// Determines whether `Apply` called with `sub_value_pointer` does not depend
  /// on `existing`.
  ///
  /// If this returns `true`, `Apply` may be called with a dummy value
  /// (e.g. `discarded`) as `existing`.
  bool CanApplyUnconditionally(std::string_view sub_value_pointer) const;

  /// Adds a change to the map.
  ///
  /// \param sub_value_pointer JSON Pointer specifying path to modify.
  /// \param sub_value New value to set for `sub_value_pointer`.
  /// \error `absl::StatusCode::kFailedPrecondition` if `sub_value_pointer` is
  ///     not compatible with an existing change.
  absl::Status AddChange(std::string_view sub_value_pointer,
                         ::nlohmann::json sub_value);

  /// Returns the underlying map representation.
  const Map& underlying_map() const { return map_; }

 private:
  Map map_;
};

}  // namespace internal_json_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_JSON_JSON_CHANGE_MAP_H_
