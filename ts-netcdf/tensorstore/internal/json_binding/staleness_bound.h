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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_STALENESS_BOUND_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_STALENESS_BOUND_H_

/// \file
///
/// Defines a JSON binder for `StalenessBound`.

#include "absl/status/status.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/staleness_bound.h"

namespace tensorstore {
namespace internal {

/// JSON binder for `StalenessBound`.
///
/// See :json-schema:`CacheRevalidationBound`.
TENSORSTORE_DECLARE_JSON_BINDER(StalenessBoundJsonBinder, StalenessBound);

}  // namespace internal
namespace internal_json_binding {
template <>
inline constexpr auto DefaultBinder<StalenessBound> =
    internal::StalenessBoundJsonBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_STALENESS_BOUND_H_
