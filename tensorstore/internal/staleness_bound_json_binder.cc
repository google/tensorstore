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

#include "tensorstore/internal/staleness_bound_json_binder.h"

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/staleness_bound.h"

namespace tensorstore {
namespace internal {

TENSORSTORE_DEFINE_JSON_BINDER(
    StalenessBoundJsonBinder,
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
      if constexpr (is_loading) {
        if (const auto* b = j->get_ptr<const bool*>()) {
          *obj = *b ? absl::InfiniteFuture() : absl::InfinitePast();
        } else if (j->is_number()) {
          const double t = static_cast<double>(*j);
          *obj = absl::UnixEpoch() + absl::Seconds(t);
        } else if (*j == "open") {
          static_cast<absl::Time&>(*obj) = absl::InfiniteFuture();
          obj->bounded_by_open_time = true;
        } else {
          return internal_json::ExpectedError(*j,
                                              "boolean, number, or \"open\"");
        }
      } else {
        if (obj->bounded_by_open_time) {
          *j = "open";
        } else {
          const absl::Time& t = *obj;
          if (t == absl::InfiniteFuture()) {
            *j = true;
          } else if (t == absl::InfinitePast()) {
            *j = false;
          } else {
            *j = absl::ToDoubleSeconds(t - absl::UnixEpoch());
          }
        }
      }
      return absl::OkStatus();
    })

}  // namespace internal
}  // namespace tensorstore
