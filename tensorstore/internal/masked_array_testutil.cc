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

#include "tensorstore/internal/masked_array_testutil.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_elementwise_input_transform.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

absl::Status WriteToMaskedArray(SharedOffsetArray<void> output, MaskData* mask,
                                IndexTransformView<> input_to_output,
                                const NDIterable& source, Arena* arena) {
  const DimensionIndex output_rank = output.rank();
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dest_iterable,
      GetTransformedArrayNDIterable(output, input_to_output, arena));

  TENSORSTORE_RETURN_IF_ERROR(NDIterableCopier(source, *dest_iterable,
                                               input_to_output.input_shape(),
                                               arena)
                                  .Copy());
  DimensionIndex layout_order[kMaxRank];
  tensorstore::span<DimensionIndex> layout_order_span(layout_order,
                                                      output_rank);
  SetPermutationFromStrides(output.byte_strides(), layout_order_span);
  WriteToMask(mask, output.domain(), input_to_output,
              ContiguousLayoutPermutation<>(layout_order_span), arena);
  return absl::OkStatus();
}

absl::Status WriteToMaskedArray(SharedOffsetArray<void> output, MaskData* mask,
                                IndexTransformView<> input_to_output,
                                TransformedArray<const void> source,
                                ElementCopyFunction::Closure copy_function) {
  if (source.domain().box() != input_to_output.domain().box()) {
    return absl::InvalidArgumentError(
        "Source domain does not match masked array");
  }
  unsigned char arena_buffer[48 * 1024];
  Arena arena(arena_buffer);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto source_iterable,
      GetTransformedArrayNDIterable(UnownedToShared(source), &arena));
  auto transformed_source_iterable = GetElementwiseInputTransformNDIterable(
      {{std::move(source_iterable)}}, output.dtype(), copy_function, &arena);
  return WriteToMaskedArray(std::move(output), mask, input_to_output,
                            *transformed_source_iterable, &arena);
}

}  // namespace internal
}  // namespace tensorstore
