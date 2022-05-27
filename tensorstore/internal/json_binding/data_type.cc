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

#include "tensorstore/internal/json_binding/data_type.h"

#include <string>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

TENSORSTORE_DEFINE_JSON_BINDER(DataTypeJsonBinder, [](auto is_loading,
                                                      const auto& options,
                                                      auto* obj,
                                                      ::nlohmann::json* j) {
  if constexpr (is_loading) {
    return internal_json_binding::Compose<std::string>(
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
})

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
                               })

TENSORSTORE_DEFINE_JSON_BINDER(
    ConstrainedDataTypeJsonBinder,
    [](auto is_loading, const auto& options, auto* obj, ::nlohmann::json* j) {
      return Validate(
          [](const auto& options, DataType* d) {
            if (options.dtype().valid() && d->valid() &&
                options.dtype() != *d) {
              return absl::InvalidArgumentError(
                  tensorstore::StrCat("Expected data type of ", options.dtype(),
                                      " but received: ", *d));
            }
            return absl::OkStatus();
          },
          DefaultValue([dtype = options.dtype()](DataType* d) { *d = dtype; }))(
          is_loading, options, obj, j);
    })

}  // namespace internal_json_binding
}  // namespace tensorstore
