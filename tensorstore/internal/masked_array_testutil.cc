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
#include "tensorstore/data_type.h"
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
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

absl::Status WriteToMaskedArray(ElementPointer<void> output_ptr, MaskData* mask,
                                BoxView<> output_box,
                                IndexTransformView<> input_to_output,
                                const NDIterable& source, Arena* arena) {
  const DimensionIndex output_rank = output_box.rank();
  Index data_byte_strides_storage[kMaxRank];
  const span<Index> data_byte_strides(&data_byte_strides_storage[0],
                                      output_rank);
  ComputeStrides(ContiguousLayoutOrder::c, output_ptr.dtype()->size,
                 output_box.shape(), data_byte_strides);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dest_iterable,
      GetTransformedArrayNDIterable(
          {UnownedToShared(AddByteOffset(
               output_ptr, -IndexInnerProduct(output_box.origin(),
                                              span(data_byte_strides)))),
           StridedLayoutView<dynamic_rank, offset_origin>(output_box,
                                                          data_byte_strides)},
          input_to_output, arena));

  TENSORSTORE_RETURN_IF_ERROR(NDIterableCopier(source, *dest_iterable,
                                               input_to_output.input_shape(),
                                               arena)
                                  .Copy());
  WriteToMask(mask, output_box, input_to_output, arena);
  return absl::OkStatus();
}

absl::Status WriteToMaskedArray(ElementPointer<void> output_ptr, MaskData* mask,
                                BoxView<> output_box,
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
      {{std::move(source_iterable)}}, output_ptr.dtype(), copy_function,
      &arena);
  return WriteToMaskedArray(output_ptr, mask, output_box, input_to_output,
                            *transformed_source_iterable, &arena);
}

}  // namespace internal
}  // namespace tensorstore
