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

#include "tensorstore/spec.h"

#include "tensorstore/driver/driver.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/json.h"

namespace tensorstore {

Result<Spec> Spec::Convert(const SpecRequestOptions& options) const {
  if (!impl_.driver_spec) {
    return *this;
  }
  Spec new_spec;
  new_spec.impl_.transform_spec = impl_.transform_spec;
  TENSORSTORE_ASSIGN_OR_RETURN(new_spec.impl_.driver_spec,
                               impl_.driver_spec->Convert(options));
  return new_spec;
}

std::ostream& operator<<(std::ostream& os, const Spec& spec) {
  return os << ::nlohmann::json(spec).dump();
}

bool operator==(const Spec& a, const Spec& b) {
  auto result_a = a.ToJson(tensorstore::IncludeContext{true});
  auto result_b = b.ToJson(tensorstore::IncludeContext{true});
  if (!result_a || !result_b) return false;
  return *result_a == *result_b;
}

Status ValidateSpecRankConstraint(DimensionIndex actual_rank,
                                  DimensionIndex rank_constraint) {
  if (IsRankExplicitlyConvertible(actual_rank, rank_constraint)) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      StrCat("Expected TensorStore of rank ", rank_constraint,
             " but received TensorStore of rank ", actual_rank));
}

namespace jb = tensorstore::internal::json_binding;
TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    Spec,
    jb::Projection(&Spec::impl_, internal::TransformedDriverSpecJsonBinder))

}  // namespace tensorstore
