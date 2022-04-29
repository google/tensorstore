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
#include <new>
#include <utility>

#include "absl/container/fixed_array.h"
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
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_elementwise_input_transform.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

MaskedArrayWriteResult WriteToMaskedArray(ElementPointer<void> output_ptr,
                                          MaskData* mask, BoxView<> output_box,
                                          IndexTransformView<> input_to_output,
                                          const NDIterable& source,
                                          Arena* arena) {
  const DimensionIndex output_rank = output_box.rank();
  absl::FixedArray<Index, kNumInlinedDims> data_byte_strides(output_rank);
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
          input_to_output, arena),
      (MaskedArrayWriteResult{_, false}));

  NDIterableCopier copier(source, *dest_iterable, input_to_output.input_shape(),
                          arena);
  auto copy_status = copier.Copy();
  return {std::move(copy_status),
          WriteToMask(mask, output_box, input_to_output,
                      copier.layout_info().layout_view(),
                      copier.stepper().position(), arena)};
}

MaskedArrayWriteResult WriteToMaskedArray(
    ElementPointer<void> output_ptr, MaskData* mask, BoxView<> output_box,
    IndexTransformView<> input_to_output, TransformedArray<const void> source,
    ElementCopyFunction::Closure copy_function) {
  if (source.domain().box() != input_to_output.domain().box()) {
    return MaskedArrayWriteResult{
        absl::InvalidArgumentError("Source domain does not match masked array"),
        false};
  }
  unsigned char arena_buffer[48 * 1024];
  Arena arena(arena_buffer);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto source_iterable,
      GetTransformedArrayNDIterable(UnownedToShared(source), &arena),
      (MaskedArrayWriteResult{_, false}));
  auto transformed_source_iterable = GetElementwiseInputTransformNDIterable(
      {{std::move(source_iterable)}}, output_ptr.dtype(), copy_function,
      &arena);
  return WriteToMaskedArray(output_ptr, mask, output_box, input_to_output,
                            *transformed_source_iterable, &arena);
}

}  // namespace internal
}  // namespace tensorstore
