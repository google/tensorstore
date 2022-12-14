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

#include "tensorstore/index_space/index_vector_or_scalar.h"

#include <system_error>  // NOLINT

#include "absl/status/status.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

absl::Status CheckIndexVectorSize(IndexVectorOrScalarView indices,
                                  DimensionIndex size) {
  if (indices.pointer && indices.size_or_scalar != size)
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of dimensions (", size, ") does not match number of indices (",
        indices.size_or_scalar, ")"));
  return absl::OkStatus();
}

}  // namespace internal_index_space
}  // namespace tensorstore
