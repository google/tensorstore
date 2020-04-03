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

#include "tensorstore/index_space/internal/translate_op.h"
#include "absl/container/fixed_array.h"
#include "tensorstore/internal/integer_overflow.h"

namespace tensorstore {
namespace internal_index_space {

namespace {
/// For each output dimension `output_dim`:
///
///   If the mapping method is `single_input_dimension`: adjusts
///     `output_index_maps[output_dim].offset()` to account for the shift in the
///     input domain specified by `input_offsets[input_dim]`.
///
///   If the mapping method is `array`: adjusts the element pointer to account
///     for the shift in the input domain.
///
/// This is a helper function used by `ApplyTranslate`.
Status TranslateOutputOffsetsUsingInputOffsets(TransformRep* transform,
                                               const Index* input_offsets) {
  const DimensionIndex output_rank = transform->output_rank;
  const DimensionIndex input_rank = transform->input_rank;
  span<OutputIndexMap> maps = transform->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        const Index offset_change = input_offsets[input_dim];
        Index new_offset;
        if (internal::MulOverflow(offset_change, map.stride(), &new_offset) ||
            internal::SubOverflow(map.offset(), new_offset, &map.offset())) {
          return absl::InvalidArgumentError(
              StrCat("Integer overflow computing output offset for dimension ",
                     output_dim, "."));
        }
        break;
      }
      case OutputIndexMethod::array: {
        auto& index_array_data = map.index_array_data();
        index_array_data.element_pointer = AddByteOffset(
            std::move(index_array_data.element_pointer),
            -IndexInnerProduct(input_rank, index_array_data.byte_strides,
                               input_offsets));
        break;
      }
      case OutputIndexMethod::constant:
        break;
    }
  }
  return absl::OkStatus();
}
}  // namespace

Result<IndexTransform<>> ApplyTranslate(IndexTransform<> transform,
                                        DimensionIndexBuffer* dimensions,
                                        IndexVectorOrScalar offsets,
                                        bool translate_to) {
  const DimensionIndex num_dims = dimensions->size();
  const DimensionIndex input_rank = transform.input_rank();
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(offsets, num_dims));
  TransformRep::Ptr<> rep =
      MutableRep(TransformAccess::rep_ptr<container>(std::move(transform)));
  const auto input_domain = rep->input_domain(input_rank);

  // Maps input dimensions to the corresponding offset in `offsets`.
  absl::FixedArray<Index, internal::kNumInlinedDims> input_offsets(input_rank,
                                                                   0);
  // Shift the input domain.
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    const DimensionIndex input_dim = (*dimensions)[i];
    Index offset = offsets[i];
    if (offset == kImplicit) continue;
    const auto old_interval = input_domain[input_dim];
    TENSORSTORE_ASSIGN_OR_RETURN(IndexInterval new_interval,
                                 translate_to
                                     ? ShiftIntervalTo(old_interval, offset)
                                     : ShiftInterval(old_interval, offset));
    if (translate_to) offset -= old_interval.inclusive_min();
    input_domain[input_dim] = new_interval;
    input_offsets[input_dim] = offset;
  }
  TENSORSTORE_RETURN_IF_ERROR(
      TranslateOutputOffsetsUsingInputOffsets(rep.get(), input_offsets.data()));
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
