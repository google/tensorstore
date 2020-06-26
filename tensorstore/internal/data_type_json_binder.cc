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

#include "tensorstore/internal/data_type_json_binder.h"

#include "absl/status/status.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {
namespace json_binding {

TENSORSTORE_DEFINE_JSON_BINDER(
    DataTypeJsonBinder,
    [](auto is_loading, const auto& options, auto* obj, ::nlohmann::json* j) {
      if constexpr (is_loading) {
        return json_binding::Compose<std::string>(
            [](auto is_loading, const auto& options, DataType* obj, auto* id) {
              *obj = tensorstore::GetDataType(*id);
              if (!obj->valid()) {
                return absl::Status(
                    absl::StatusCode::kInvalidArgument,
                    tensorstore::StrCat("Unsupported data type: ",
                                        tensorstore::QuoteString(*id)));
              }
              return absl::OkStatus();
            })(is_loading, options, obj, j);
      } else {
        if (!obj->valid()) {
          *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
        } else if (obj->id() == DataTypeId::custom) {
          return absl::Status(absl::StatusCode::kInvalidArgument,
                              "Data type has no canonical identifier");
        } else {
          *j = obj->name();
        }
        return absl::OkStatus();
      }
    });

TENSORSTORE_DEFINE_JSON_BINDER(OptionalDataTypeJsonBinder,
                               [](auto is_loading, const auto& options,
                                  auto* obj, ::nlohmann::json* j) {
                                 if constexpr (is_loading) {
                                   if (j->is_discarded()) {
                                     *obj = DataType{};
                                     return absl::OkStatus();
                                   }
                                 }
                                 return DataTypeJsonBinder(is_loading, options,
                                                           obj, j);
                               });

}  // namespace json_binding
}  // namespace internal
}  // namespace tensorstore
