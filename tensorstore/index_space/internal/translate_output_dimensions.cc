// Copyright 2023 The TensorStore Authors
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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {

Result<TransformRep::Ptr<>> TranslateOutputDimensionsBy(
    TransformRep::Ptr<> transform, span<const Index> offsets) {
  assert(transform);
  if (offsets.size() != transform->output_rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot translate output dimensions of rank %d -> %d transform with "
        "offset vector of length %d",
        transform->input_rank, transform->output_rank, offsets.size()));
  }
  transform = MutableRep(std::move(transform), /*domain_only=*/false);
  for (DimensionIndex output_dim = 0; output_dim < offsets.size();
       ++output_dim) {
    auto& map = transform->output_index_maps()[output_dim];
    map.offset() =
        internal::wrap_on_overflow::Add(map.offset(), offsets[output_dim]);
  }
  return transform;
}

}  // namespace internal_index_space
}  // namespace tensorstore
