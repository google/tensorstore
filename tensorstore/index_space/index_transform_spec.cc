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

#include "tensorstore/index_space/index_transform_spec.h"

#include <ostream>
#include <string>

#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

namespace internal_index_space {

absl::Status UnspecifiedTransformError() {
  return absl::InvalidArgumentError("Transform is unspecified");
}

}  // namespace internal_index_space

std::ostream& operator<<(std::ostream& os, const IndexTransformSpec& s) {
  if (s.transform_.valid()) {
    os << s.transform_;
  } else {
    os << s.input_rank_;
  }
  return os;
}

Result<IndexTransformSpec> ComposeIndexTransformSpecs(
    IndexTransformSpec b_to_c, IndexTransformSpec a_to_b) {
  if (!IsRankExplicitlyConvertible(b_to_c.input_rank(), a_to_b.output_rank())) {
    return Status(
        absl::StatusCode::kInvalidArgument,
        StrCat("Cannot compose transform of rank ", b_to_c.input_rank(), " -> ",
               b_to_c.output_rank(), " with transform of rank ",
               a_to_b.input_rank(), " -> ", a_to_b.output_rank()));
  }
  if (b_to_c.transform().valid()) {
    if (a_to_b.transform().valid()) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto a_to_c, ComposeTransforms(std::move(b_to_c).transform(),
                                         std::move(a_to_b).transform()));
      return IndexTransformSpec{std::move(a_to_c)};
    }
    return b_to_c;
  } else if (a_to_b.transform().valid()) {
    return a_to_b;
  } else {
    // Due to check above, `a_to_b.input_rank()` or `b_to_c.input_rank()` must
    // be equal or at least one must be `dynamic_rank`.  Return the
    // non-`dynamic_rank` value, if any.
    return IndexTransformSpec(
        std::max(a_to_b.input_rank(), b_to_c.input_rank()));
  }
}

}  // namespace tensorstore
