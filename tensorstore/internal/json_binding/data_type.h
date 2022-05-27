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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_DATA_TYPE_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_DATA_TYPE_H_

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal_json_binding {

/// DataType JSON Binder that requires a valid data type.
TENSORSTORE_DECLARE_JSON_BINDER(DataTypeJsonBinder, DataType)

/// DataType JSON Binder that allows the data type to be unspecified (via
/// `::nlohmann::json::value_discarded_t`).
TENSORSTORE_DECLARE_JSON_BINDER(OptionalDataTypeJsonBinder, DataType)

/// DataType JSON Binder where `options.dtype` specifies both a constraint
/// and a default value.
TENSORSTORE_DECLARE_JSON_BINDER(ConstrainedDataTypeJsonBinder, DataType,
                                JsonSerializationOptions,
                                JsonSerializationOptions)

template <>
inline constexpr auto DefaultBinder<DataType> = OptionalDataTypeJsonBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_DATA_TYPE_H_
