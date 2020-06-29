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

#include "tensorstore/index_space/internal/single_index_slice_op.h"

#include "absl/container/fixed_array.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/integer_overflow.h"

namespace tensorstore {
namespace internal_index_space {
namespace {
/// Specifies information specific to a single input dimension in the original
/// transform (prior to applying the index slicing operation).
struct InputDimensionSingletonSliceInfo {
  /// Specifies the input dimension in the new transform corresponding to this
  /// input dimension in the original transform.  Equals `-1` if there is no
  /// corresponding input dimension in the new transform, because this is one of
  /// the selected dimensions.
  DimensionIndex new_input_dim;

  /// If `new_input_dim == -1` (i.e. this is one of the dimensions being
  /// sliced), specifies the slice index.  Otherwise, this field is ignored.
  Index offset;
};

struct SingletonSlicingInfo {
  explicit SingletonSlicingInfo(DimensionIndex original_input_rank,
                                DimensionIndex new_input_rank)
      : original_input_dimension_info(original_input_rank,
                                      InputDimensionSingletonSliceInfo{0, 0}),
        new_input_rank(new_input_rank) {}

  absl::FixedArray<InputDimensionSingletonSliceInfo, internal::kNumInlinedDims>
      original_input_dimension_info;
  DimensionIndex new_input_rank;
};

/// Validates the specified indices, and computes the `SingletonSlicingInfo`
/// representation of the slicing operation.
///
/// \param original Non-null pointer to the original transform.
/// \param dimension_buffer[in,out] Must be non-null.  On input, specifies the
///     dimensions to be sliced.  On successful return, cleared.
/// \param indices The vector of indices of length `dimension_buffer->size()`
///     specifying the index to slice from each selected dimension.
/// \returns The slicing information.
Result<SingletonSlicingInfo> GetSingletonSlicingInfo(
    TransformRep* original, DimensionIndexBuffer* dimensions_buffer,
    IndexVectorOrScalar indices) {
  const span<const DimensionIndex> dimensions(*dimensions_buffer);
  const DimensionIndex num_dims = dimensions.size();
  const DimensionIndex original_input_rank = original->input_rank;
  const DimensionIndex new_input_rank = original_input_rank - num_dims;
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(indices, num_dims));

  // Declare the return value here so that named return value optimization
  // (NRVO) is possible.
  Result<SingletonSlicingInfo> result(tensorstore::in_place,
                                      original_input_rank, new_input_rank);
  const Index* indices_pointer =
      indices.pointer ? indices.pointer : &indices.size_or_scalar;
  const Index indices_stride = indices.pointer ? 1 : 0;
  // For each existing dimension being sliced in `dimensions`, validate the
  // specified slice index and update the corresponding
  // InputDimensionSingletonSliceInfo structure, which also marks the existing
  // dimension as being one of the sliced dimensions.
  std::string slice_error;
  for (DimensionIndex i = 0; i < num_dims; ++i) {
    const DimensionIndex original_input_dim = dimensions[i];
    const Index index = indices_pointer[i * indices_stride];
    const auto domain = original->input_dimension(original_input_dim)
                            .optionally_implicit_domain();
    if (!Contains(domain.effective_interval(), index)) {
      StrAppend(&slice_error, (slice_error.empty() ? "" : ", "),
                "in input dimension ", original_input_dim, " index ", index,
                " is outside valid domain ", domain);
    }
    result->original_input_dimension_info[original_input_dim] =
        InputDimensionSingletonSliceInfo{-1, index};
  }
  if (!slice_error.empty()) {
    // Assign to result, rather than just returning the error, to encourage
    // NRVO.
    result = absl::OutOfRangeError(StrCat("Slice mismatch: ", slice_error));
    return result;
  }

  // For each existing dimension not being sliced, set the corresponding the
  // InputDimensionSingletonSliceInfo structure to indicate the mapping from
  // original input dimension index to new input dimension index.
  for (DimensionIndex original_input_dim = 0, new_input_dim = 0;
       original_input_dim < original_input_rank; ++original_input_dim) {
    auto& new_dim =
        result->original_input_dimension_info[original_input_dim].new_input_dim;
    if (new_dim == -1) continue;
    new_dim = new_input_dim;
    ++new_input_dim;
  }
  // Modify the dimensions buffer to reflect the new (empty) dimension selection
  // after the slicing operation is performed.
  dimensions_buffer->clear();
  return result;
}

/// Assigns `new_transform` to the result of applying a singleton slice
/// operation on `original_transform`.
///
/// \param original_transform The original transform prior to the slicing
///     operation.
/// \param new_transform The new transform after the slicing operation.  May
///     alias `original_transform`.
/// \param original_input_dimension_info A pointer to the array returned from
///     `GetSingletonSlicingInfo`.
/// \returns `Status()` on success, or `absl::StatusCode::kInvalidArgument` if
///     an integer overflow occurs.
Status PerformSingleIndexSlice(TransformRep* original_transform,
                               TransformRep* new_transform,
                               const SingletonSlicingInfo& info) {
  const DimensionIndex original_input_rank = original_transform->input_rank;
  const DimensionIndex new_input_rank = info.new_input_rank;
  span<const InputDimensionSingletonSliceInfo> original_input_dimension_info =
      info.original_input_dimension_info;
  // Indicates whether the new transform has a zero-element transform.
  bool has_zero_elements = false;
  // Set the fields of each input dimension of the new transform from the
  // corresponding fields of the original input dimension.
  for (DimensionIndex original_input_dim = 0, new_input_dim = 0;
       original_input_dim < original_input_rank; ++original_input_dim) {
    if (original_input_dimension_info[original_input_dim].new_input_dim < 0)
      continue;
    const InputDimensionRef new_dim_ref =
        new_transform->input_dimension(new_input_dim);
    new_dim_ref = original_transform->input_dimension(original_input_dim);
    if (new_dim_ref.domain().empty()) has_zero_elements = true;
    ++new_input_dim;
  }
  const DimensionIndex output_rank = original_transform->output_rank;
  span<const OutputIndexMap> original_maps =
      original_transform->output_index_maps().first(output_rank);
  span<OutputIndexMap> new_maps =
      new_transform->output_index_maps().first(output_rank);
  // Compute the output index maps for the new transform.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const OutputIndexMap& original_map = original_maps[output_dim];
    OutputIndexMap& new_map = new_maps[output_dim];
    switch (original_map.method()) {
      case OutputIndexMethod::constant: {
        new_map.offset() = original_map.offset();
        new_map.SetConstant();
        new_map.stride() = 0;
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex original_input_dim =
            original_map.input_dimension();
        ABSL_ASSERT(original_input_dim >= 0 &&
                    original_input_dim < original_input_rank);
        const auto slice_info =
            original_input_dimension_info[original_input_dim];
        const Index output_stride = original_map.stride();
        const Index output_offset = original_map.offset();
        if (slice_info.new_input_dim == -1) {
          // The input dimension is being sliced, which makes this into a
          // constant mapping.
          Index new_offset;
          if (internal::MulOverflow(slice_info.offset, output_stride,
                                    &new_offset) ||
              internal::AddOverflow(new_offset, output_offset,
                                    &new_map.offset())) {
            return absl::InvalidArgumentError(StrCat(
                "Integer overflow computing offset for output dimension ",
                output_dim, "."));
          }
          new_map.SetConstant();
          new_map.stride() = 0;
        } else {
          // The input dimension is not being sliced.
          new_map.SetSingleInputDimension(slice_info.new_input_dim);
          new_map.stride() = output_stride;
          new_map.offset() = output_offset;
        }
        break;
      }
      case OutputIndexMethod::array: {
        const IndexArrayData& original_index_array_data =
            original_map.index_array_data();
        IndexArrayData& new_index_array_data =
            new_map.SetArrayIndexing(new_input_rank);
        new_index_array_data.index_range =
            original_index_array_data.index_range;
        Index array_byte_offset = 0;
        // Indicates whether the new index array has any non-zero byte strides
        // (i.e. contains more than one distinct element).
        bool has_non_zero_byte_strides = false;
        // Compute the base pointer and byte strides of the new index array.
        for (DimensionIndex original_input_dim = 0;
             original_input_dim < original_input_rank; ++original_input_dim) {
          const auto slice_info =
              original_input_dimension_info[original_input_dim];
          const Index byte_stride =
              original_index_array_data.byte_strides[original_input_dim];
          if (slice_info.new_input_dim == -1) {
            array_byte_offset = internal::wrap_on_overflow::Add(
                array_byte_offset, internal::wrap_on_overflow::Multiply(
                                       byte_stride, slice_info.offset));
          } else {
            new_index_array_data.byte_strides[slice_info.new_input_dim] =
                byte_stride;
            if (byte_stride != 0) has_non_zero_byte_strides = true;
          }
        }
        Index output_stride = original_map.stride();
        Index output_offset = original_map.offset();
        if (has_non_zero_byte_strides) {
          // The new index array is not a singleton array, therefore the mapping
          // must remain as an index array mapping.
          new_index_array_data.element_pointer = AddByteOffset(
              original_index_array_data.element_pointer, array_byte_offset);
        } else {
          // Index array has become rank 0, so we can replace it with a constant
          // index.
          if (!has_zero_elements) {
            TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
                original_index_array_data.element_pointer
                    .byte_strided_pointer()[array_byte_offset],
                new_index_array_data.index_range, &output_offset,
                &output_stride));
          } else {
            output_offset = 0;
            output_stride = 0;
          }
          new_map.SetConstant();
        }
        new_map.stride() = output_stride;
        new_map.offset() = output_offset;
        break;
      }
    }
  }
  new_transform->input_rank = new_input_rank;
  new_transform->output_rank = output_rank;
  return absl::OkStatus();
}
}  // namespace

Result<IndexTransform<>> ApplySingleIndexSlice(IndexTransform<> transform,
                                               DimensionIndexBuffer* dimensions,
                                               IndexVectorOrScalar indices) {
  TransformRep* rep = TransformAccess::rep(transform);
  auto slicing_info = GetSingletonSlicingInfo(rep, dimensions, indices);
  if (!slicing_info) return slicing_info.status();
  auto new_rep =
      NewOrMutableRep(rep, slicing_info->new_input_rank, rep->output_rank);
  TENSORSTORE_RETURN_IF_ERROR(
      PerformSingleIndexSlice(rep, new_rep.get(), *slicing_info));
  return TransformAccess::Make<IndexTransform<>>(new_rep);
}

}  // namespace internal_index_space
}  // namespace tensorstore
