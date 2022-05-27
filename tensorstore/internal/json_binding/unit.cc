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

#include "tensorstore/internal/json_binding/unit.h"

#include <cmath>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_tuple.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal_json_binding {

TENSORSTORE_DEFINE_JSON_BINDER(
    UnitJsonBinder,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      if constexpr (is_loading) {
        if (auto* s = j->get_ptr<const std::string*>()) {
          *obj = Unit(*s);
          return absl::OkStatus();
        } else if (j->is_number()) {
          // Unit-less number
          *obj = Unit(j->get<double>(), "");
          return absl::OkStatus();
        }
      }
      return HeterogeneousArray(
          Projection<&Unit::multiplier>(Validate([](const auto& options,
                                                    auto* num) {
            if (*num > 0 && std::isfinite(*num)) return absl::OkStatus();
            return internal_json::ExpectedError(*num, "finite positive number");
          })),
          Projection<&Unit::base_unit>())(is_loading, options, obj, j);
    });

}  // namespace internal_json_binding
}  // namespace tensorstore
