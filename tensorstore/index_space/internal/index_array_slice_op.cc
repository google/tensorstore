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

#include "tensorstore/index_space/internal/index_array_slice_op.h"

#include <numeric>

#include "absl/container/fixed_array.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

namespace {
bool BroadcastSizes(Index source, Index* result) {
  if (source == *result) return true;
  if (*result == 1) {
    *result = source;
    return true;
  } else if (source == 1) {
    return true;
  }
  return false;
}

/// Updates `result_shape` to be the result of broadcasting `source_shape` and
/// `result_shape`.
///
/// \returns `true` if broadcasting is successful, `false` otherwise.
bool BroadcastShapes(span<const Index> source_shape, span<Index> result_shape) {
  if (source_shape.size() != result_shape.size()) return false;
  for (DimensionIndex i = 0; i < source_shape.size(); ++i) {
    if (!BroadcastSizes(source_shape[i], &result_shape[i])) return false;
  }
  return true;
}

/// Computes a transform where the specified `dimensions` are jointly indexed by
/// the specified index arrays.
///
/// This is a helper function used by MakeTransformFromIndexArrays and
/// MakeTransformFromIndexVectorArray.
///
/// \param num_new_dims The number of new dimensions, corresponding to the
///     common rank of the index arrays.
/// \param orig_transform The original transform to which this transform will be
///     applied.  Input dimension fields for input dimensions of the returned
///     transform that correspond to dimensions of `orig_transform` are copied
///     from the corresponding fields.
/// \param dimensions Must be non-null.  On input, specifies the output
///     dimensions to be mapped using index arrays.  On return, specifies the
///     input dimensions corresponding to the common domain of the index arrays.
/// \param get_new_dimension_bounds Function with signature:
///     `(DimensionIndex new_dim) -> IndexInterval`
///     that returns the bounds of the specified new dimension in the common
///     domain of the index arrays.
/// \param get_index_array_base_pointer Function with signature:
///     `(DimensionIndex indexed_dim) -> std::shared_ptr<const Index>` that
///     returns the index array base pointer for the specified indexed
///     dimension; `indexed_dim` is an index into the `*dimensions` sequence.
/// \param get_index_array_byte_stride Function with signature: `(DimensionIndex
///     indexed_dim, DimensionIndex new_dim) -> Index` that returns the byte
///     stride for the dimension `new_dim` of the `indexed_dim` index array.
/// \returns The transform where the output dimensions specified in
///     `*dimensions` are mapped using the specified index arrays, and the
///     remaining output dimensions are identity mapped.
/// \error `absl::StatusCode::kInvalidArgument` if the resultant input rank is
///     invalid.
template <typename GetNewDimensionShapeFn, typename GetIndexArrayBasePointerFn,
          typename GetIndexArrayByteStrideFn>
Result<TransformRep::Ptr<>> MakeTransformFromJointIndexArrays(
    DimensionIndex num_new_dims, TransformRep* orig_transform,
    DimensionIndexBuffer* dimensions,
    GetNewDimensionShapeFn get_new_dimension_bounds,
    GetIndexArrayBasePointerFn get_index_array_base_pointer,
    GetIndexArrayByteStrideFn get_index_array_byte_stride) {
  const DimensionIndex num_indexed_dims = dimensions->size();
  const DimensionIndex output_rank = orig_transform->input_rank;
  const DimensionIndex input_rank =
      output_rank - dimensions->size() + num_new_dims;
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(input_rank));
  auto result = TransformRep::Allocate(input_rank, output_rank);
  result->input_rank = input_rank;
  result->output_rank = output_rank;
  // Set all bounds to explicit.  These defaults are only used for dimensions
  // corresponding to the domain of the index arrays.  For identity-mapped
  // dimensions, the defaults are overridden.
  result->implicit_lower_bounds = false;
  result->implicit_upper_bounds = false;
  span<OutputIndexMap> maps = result->output_index_maps().first(output_rank);
  const DimensionIndex num_preserved_dims = output_rank - num_indexed_dims;
  // Set all output dimensions to single_input_dimension index method.  The
  // output dimensions corresponding to the index array dimensions will then be
  // set to either constant or array index method.  All output dimensions not
  // corresponding to the index array dimensions will remain set to the
  // single_input_dimension index method, which allows them to be distinguished.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    maps[output_dim].SetSingleInputDimension(0);
  }
  const auto input_domain = result->input_domain(input_rank);
  // Sets the input domain for dimensions corresponding to the domain of the
  // index arrays.
  for (DimensionIndex new_dim = 0; new_dim < num_new_dims; ++new_dim) {
    input_domain[new_dim] = get_new_dimension_bounds(new_dim);
  }
  // Sets all array-indexed dimensions to have array output index maps.
  for (DimensionIndex indexed_dim = 0; indexed_dim < num_indexed_dims;
       ++indexed_dim) {
    const DimensionIndex output_dim = (*dimensions)[indexed_dim];
    auto& map = maps[output_dim];
    map.offset() = 0;
    map.stride() = 1;
    auto& index_array_data = map.SetArrayIndexing(input_rank);
    std::fill_n(index_array_data.byte_strides + num_new_dims,
                num_preserved_dims, 0);
    for (DimensionIndex new_dim = 0; new_dim < num_new_dims; ++new_dim) {
      index_array_data.byte_strides[new_dim] =
          get_index_array_byte_stride(indexed_dim, new_dim);
    }
    index_array_data.element_pointer =
        get_index_array_base_pointer(indexed_dim);
  }
  // Sets the output index maps for output dimensions not indexed by the index
  // array to be identity maps, and copies the input dimension fields from the
  // corresponding dimension of `orig_transform`.
  for (DimensionIndex output_dim = 0, input_dim = num_new_dims;
       output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    map.SetSingleInputDimension(input_dim);
    map.offset() = 0;
    map.stride() = 1;
    result->input_dimension(input_dim) =
        orig_transform->input_dimension(output_dim);
    ++input_dim;
  }
  if (IsDomainExplicitlyEmpty(result.get())) {
    ReplaceAllIndexArrayMapsWithConstantMaps(result.get());
  }
  dimensions->resize(num_new_dims);
  std::iota(dimensions->begin(), dimensions->end(),
            static_cast<DimensionIndex>(0));
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

Result<TransformRep::Ptr<>> MakeTransformFromIndexArrays(
    TransformRep* orig_transform, DimensionIndexBuffer* dimensions,
    span<const SharedArrayView<const Index>> index_arrays) {
  const DimensionIndex num_indexed_dims = dimensions->size();
  if (index_arrays.size() != num_indexed_dims) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of selected dimensions (", num_indexed_dims,
        ") does not equal number of index arrays (", index_arrays.size(), ")"));
  }
  if (index_arrays.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("At least one index array must be specified"));
  }
  absl::FixedArray<Index, internal::kNumInlinedDims> shape(
      index_arrays[0].rank(), 1);

  bool error = false;
  for (DimensionIndex i = 0; i < index_arrays.size(); ++i) {
    if (!BroadcastShapes(index_arrays[i].shape(), shape)) {
      error = true;
    }
  }
  if (error) {
    std::string shape_msg;
    for (DimensionIndex i = 0; i < index_arrays.size(); ++i) {
      tensorstore::StrAppend(&shape_msg, (shape_msg.empty() ? "" : ", "),
                             index_arrays[i].shape());
    }
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Index arrays with shapes ", shape_msg,
                            " cannot be broadcast to a common shape"));
  }

  const DimensionIndex num_new_dims = shape.size();
  const auto get_new_dimension_bounds = [&](DimensionIndex new_dim) {
    return IndexInterval::UncheckedSized(0, shape[new_dim]);
  };
  const auto get_index_array_base_pointer = [&](DimensionIndex indexed_dim) {
    return index_arrays[indexed_dim].pointer();
  };
  const auto get_index_array_byte_stride = [&](DimensionIndex indexed_dim,
                                               DimensionIndex new_dim) {
    return index_arrays[indexed_dim].shape()[new_dim] == 1
               ? 0
               : index_arrays[indexed_dim].byte_strides()[new_dim];
  };
  return MakeTransformFromJointIndexArrays(
      num_new_dims, orig_transform, dimensions, get_new_dimension_bounds,
      get_index_array_base_pointer, get_index_array_byte_stride);
}

Result<TransformRep::Ptr<>> MakeTransformFromOuterIndexArrays(
    TransformRep* orig_transform, DimensionIndexBuffer* dimensions,
    span<const SharedArrayView<const Index>> index_arrays) {
  const DimensionIndex num_indexed_dims = dimensions->size();
  if (index_arrays.size() != num_indexed_dims) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Number of selected dimensions (", num_indexed_dims,
        ") does not equal number of index arrays (", index_arrays.size(), ")"));
  }
  const DimensionIndex output_rank = orig_transform->input_rank;
  DimensionIndex input_rank = output_rank - num_indexed_dims;
  for (const auto& index_array : index_arrays) {
    input_rank += index_array.rank();
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(input_rank));
  auto result = TransformRep::Allocate(input_rank, output_rank);
  result->input_rank = input_rank;
  result->output_rank = output_rank;
  // Set all bounds to explicit.  These defaults are only used for dimensions
  // corresponding to the domain of the index arrays.  For identity-mapped
  // dimensions, the defaults are overridden.
  result->implicit_lower_bounds = false;
  result->implicit_upper_bounds = false;
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      index_array_start_dim(num_indexed_dims);
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims> index_array_order(
      num_indexed_dims);
  std::iota(index_array_order.begin(), index_array_order.end(),
            static_cast<DimensionIndex>(0));
  std::sort(index_array_order.begin(), index_array_order.end(),
            [&](DimensionIndex a, DimensionIndex b) {
              return (*dimensions)[a] < (*dimensions)[b];
            });
  span<Index> input_origin = result->input_origin().first(input_rank);
  span<Index> input_shape = result->input_shape().first(input_rank);
  span<OutputIndexMap> maps = result->output_index_maps().first(output_rank);
  // Sets output index maps, input_origin, input_shape, and input_labels of
  // transform.
  for (DimensionIndex output_dim = 0, reordered_indexed_dim = 0, input_dim = 0;
       output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    map.stride() = 1;
    map.offset() = 0;
    if (reordered_indexed_dim < num_indexed_dims) {
      const DimensionIndex indexed_dim =
          index_array_order[reordered_indexed_dim];
      if ((*dimensions)[indexed_dim] == output_dim) {
        // This output dimension corresponds to an index array.
        index_array_start_dim[indexed_dim] = input_dim;
        const auto& array = index_arrays[indexed_dim];
        MutableBoxView<>(input_origin.subspan(input_dim, array.rank()),
                         input_shape.subspan(input_dim, array.rank()))
            .DeepAssign(array.domain());
        const DimensionIndex end_input_dim = input_dim + array.rank();
        if (array.num_elements() == 1) {
          map.SetConstant();
          map.offset() = *array.data();
          map.stride() = 0;
        } else {
          auto& index_array_data = map.SetArrayIndexing(input_rank);
          index_array_data.element_pointer = array.element_pointer();
          std::fill_n(index_array_data.byte_strides, input_dim, 0);
          std::copy(array.byte_strides().begin(), array.byte_strides().end(),
                    index_array_data.byte_strides + input_dim);
          std::fill(index_array_data.byte_strides + end_input_dim,
                    index_array_data.byte_strides + input_rank, 0);
        }
        input_dim = end_input_dim;
        ++reordered_indexed_dim;
        continue;
      }
    }
    // This output dimension is not array indexed.  Therefore, copy the input
    // dimension fields from the corresponding dimension of `orig_transform`.
    result->input_dimension(input_dim) =
        orig_transform->input_dimension(output_dim);
    map.SetSingleInputDimension(input_dim);
    ++input_dim;
  }
  if (IsDomainExplicitlyEmpty(result.get())) {
    ReplaceAllIndexArrayMapsWithConstantMaps(result.get());
  }
  // Sets `dimensions` to the new input dimensions corresponding to the index
  // array domains.
  dimensions->clear();
  dimensions->reserve(input_rank - output_rank);
  for (DimensionIndex indexed_dim = 0; indexed_dim < num_indexed_dims;
       ++indexed_dim) {
    const DimensionIndex start_input_dim = index_array_start_dim[indexed_dim];
    for (DimensionIndex
             input_dim = start_input_dim,
             end_input_dim = start_input_dim + index_arrays[indexed_dim].rank();
         input_dim != end_input_dim; ++input_dim) {
      dimensions->push_back(input_dim);
    }
  }
  internal_index_space::DebugCheckInvariants(result.get());
  return result;
}

Result<TransformRep::Ptr<>> MakeTransformFromIndexVectorArray(
    TransformRep* orig_transform, DimensionIndexBuffer* dimensions,
    DimensionIndex vector_dimension,
    const SharedArrayView<const Index>& index_vector_array) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      vector_dimension,
      NormalizeDimensionIndex(vector_dimension, index_vector_array.rank()));
  const DimensionIndex num_indexed_dims = dimensions->size();
  if (index_vector_array.shape()[vector_dimension] != num_indexed_dims) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Number of selected dimensions (", num_indexed_dims,
                            ") does not equal index vector length (",
                            index_vector_array.shape()[vector_dimension], ")"));
  }
  const DimensionIndex num_new_dims = index_vector_array.rank() - 1;
  const auto get_index_vector_array_dim = [&](DimensionIndex new_dim) {
    return new_dim >= vector_dimension ? new_dim + 1 : new_dim;
  };
  const auto get_new_dimension_bounds = [&](DimensionIndex new_dim) {
    return index_vector_array.domain()[get_index_vector_array_dim(new_dim)];
  };
  const auto get_index_array_base_pointer = [&](DimensionIndex indexed_dim) {
    return std::shared_ptr<const Index>(
        index_vector_array.pointer(),
        index_vector_array.byte_strided_pointer() +
            index_vector_array.byte_strides()[vector_dimension] * indexed_dim);
  };
  const auto get_index_array_byte_stride = [&](DimensionIndex indexed_dim,
                                               DimensionIndex new_dim) {
    const DimensionIndex index_vector_array_dim =
        get_index_vector_array_dim(new_dim);
    return index_vector_array.shape()[index_vector_array_dim] == 1
               ? 0
               : index_vector_array.byte_strides()[index_vector_array_dim];
  };
  return MakeTransformFromJointIndexArrays(
      num_new_dims, orig_transform, dimensions, get_new_dimension_bounds,
      get_index_array_base_pointer, get_index_array_byte_stride);
}

}  // namespace

Result<IndexTransform<>> ApplyIndexArraySlice(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    span<const SharedArrayView<const Index>> index_arrays, bool outer_indexing,
    bool domain_only) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto other_transform,
      outer_indexing
          ? MakeTransformFromOuterIndexArrays(TransformAccess::rep(transform),
                                              dimensions, index_arrays)
          : MakeTransformFromIndexArrays(TransformAccess::rep(transform),
                                         dimensions, index_arrays));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_rep,
      ComposeTransforms(TransformAccess::rep(transform),
                        /*can_move_from_b_to_c=*/false, other_transform.get(),
                        /*can_move_from_a_to_b=*/true, domain_only));
  return TransformAccess::Make<IndexTransform<>>(std::move(new_rep));
}

Result<IndexTransform<>> ApplyIndexVectorArraySlice(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    DimensionIndex vector_dimension,
    const SharedArrayView<const Index>& index_vector_array, bool domain_only) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto other_transform,
                               MakeTransformFromIndexVectorArray(
                                   TransformAccess::rep(transform), dimensions,
                                   vector_dimension, index_vector_array));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_rep,
      ComposeTransforms(TransformAccess::rep(transform),
                        /*can_move_from_b_to_c=*/false, other_transform.get(),
                        /*can_move_from_a_to_b=*/true, domain_only));
  return TransformAccess::Make<IndexTransform<>>(std::move(new_rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
