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

#include "tensorstore/index_space/internal/label_op.h"

#include <stddef.h>

#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/dimension_labels.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

Result<IndexTransform<>> ApplyLabel(IndexTransform<> transform,
                                    DimensionIndexBuffer* dimensions,
                                    internal::StringLikeSpan labels,
                                    bool domain_only) {
  if (dimensions->size() != static_cast<size_t>(labels.size())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of dimensions (", dimensions->size(),
        ") does not match number of labels (", labels.size(), ")."));
  }
  auto rep = MutableRep(
      TransformAccess::rep_ptr<container>(std::move(transform)), domain_only);
  const DimensionIndex input_rank = rep->input_rank;
  span<std::string> input_labels = rep->input_labels().first(input_rank);
  for (DimensionIndex i = 0;
       i < static_cast<DimensionIndex>(dimensions->size()); ++i) {
    const DimensionIndex input_dim = (*dimensions)[i];
    std::string_view label = labels[i];
    input_labels[input_dim].assign(label.begin(), label.end());
  }
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateDimensionLabelsAreUnique(input_labels));
  internal_index_space::DebugCheckInvariants(rep.get());
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
