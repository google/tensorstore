// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_OPTIONAL_OBJECT_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_OPTIONAL_OBJECT_H_

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_json_binding {

/// Like `Compose<::nlohmann::json::object_t>`, except that when loading,
/// `discarded` is converted to an empty object, and when saving, an empty
/// object is converted back to `discarded`.
///
/// When loading, the inner binder is always invoked.
template <typename ObjectBinder>
constexpr auto OptionalObject(ObjectBinder binder) {
  return [binder = std::move(binder)](auto is_loading, const auto& options,
                                      auto* obj, auto* j) -> absl::Status {
    ::nlohmann::json::object_t json_obj;
    if constexpr (is_loading) {
      if (!j->is_discarded()) {
        if (auto* ptr = j->template get_ptr<::nlohmann::json::object_t*>();
            ptr) {
          json_obj = std::move(*ptr);
        } else {
          return internal_json::ExpectedError(*j, "object");
        }
      }
    }
    if (auto status = internal_json_binding::Object(binder)(is_loading, options,
                                                            obj, &json_obj);
        !status.ok()) {
      return status;
    }
    if constexpr (!is_loading) {
      if (!json_obj.empty()) {
        *j = std::move(json_obj);
      } else {
        *j = ::nlohmann::json::value_t::discarded;
      }
    }
    return absl::OkStatus();
  };
}

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_OPTIONAL_OBJECT_H_
