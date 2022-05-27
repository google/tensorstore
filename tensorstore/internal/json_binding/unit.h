// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_UNIT_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_UNIT_H_

#include "absl/status/status.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal_json_binding {

/// JSON binder for `Unit`.
///
/// When converting to JSON, always uses `[multiplier, base_unit]` format.  When
/// converting from JSON, accepts the following syntax:
///
/// - `[multiplier, base_unit]`, where `multiplier` is a number and `base_unit`
///   is a string.  For example, `[5.5, "nm"]`.
///
/// - `multiplier`, where `multiplier` is a number.  For example, `5.5`.  The
///   base unit is set to `""` in this case.
///
/// - `unit`, where `unit` is a string.  For example, `"5.5"` or `"nm"` or
///   `"5.5nm"`.  The parsing of the `unit` string is done via the `Unit` string
///   constructor.
TENSORSTORE_DECLARE_JSON_BINDER(UnitJsonBinder, Unit, NoOptions,
                                IncludeDefaults);

template <>
constexpr inline auto DefaultBinder<Unit> = UnitJsonBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_UNIT_H_
