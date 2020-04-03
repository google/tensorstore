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

#include "tensorstore/index_space/internal/mark_explicit_op.h"

namespace tensorstore {
namespace internal_index_space {

Result<IndexTransform<>> ApplyChangeImplicitState(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions, bool implicit,
    bool lower, bool upper) {
  TransformRep::Ptr<> rep =
      MutableRep(TransformAccess::rep_ptr<container>(std::move(transform)));
  for (DimensionIndex input_dim : *dimensions) {
    const auto d = rep->input_dimension(input_dim);
    if (lower) d.implicit_lower_bound() = implicit;
    if (upper) d.implicit_upper_bound() = implicit;
  }
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
