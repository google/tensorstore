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

#include "tensorstore/index_space/internal/compose_transforms.h"

#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_array.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"

namespace tensorstore {
namespace internal_index_space {

namespace {

enum class ArrayUniqueElementZeroOneManyCount {
  kZero = 0,
  kOne = 1,
  kMoreThanOne = 2,
};

/// Returns a bound on the number of unique elements in an array with the
/// specified `shape` and `byte_strides`.
///
/// \param rank Rank of the array
/// \param shape Pointer to vector of length `rank`
/// \param byte_strides Pointer to vector of length `rank`
ArrayUniqueElementZeroOneManyCount GetArrayUniqueElementZeroOneManyCount(
    DimensionIndex rank, const Index* shape, const Index* byte_strides) {
  ArrayUniqueElementZeroOneManyCount result =
      ArrayUniqueElementZeroOneManyCount::kOne;
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    if (byte_strides[dim] == 0) continue;
    const Index size = shape[dim];
    if (size == 0) return ArrayUniqueElementZeroOneManyCount::kZero;
    if (size != 1) result = ArrayUniqueElementZeroOneManyCount::kMoreThanOne;
  }
  return result;
}
}  // namespace

Status ComposeTransforms(TransformRep* b_to_c, bool can_move_from_b_to_c,
                         TransformRep* a_to_b, bool can_move_from_a_to_b,
                         TransformRep* a_to_c) {
  ABSL_ASSERT(b_to_c != nullptr && a_to_b != nullptr && a_to_c != nullptr &&
              b_to_c->output_rank <= a_to_c->output_rank_capacity &&
              a_to_b->output_rank == b_to_c->input_rank &&
              a_to_b->input_rank <= a_to_c->input_rank_capacity);
  // Aliasing of `a_to_c` is not allowed unless it has maximum input/output
  // ranks of 0.
  ABSL_ASSERT(
      (a_to_c->input_rank_capacity == 0 && b_to_c->input_rank_capacity == 0) ||
      (a_to_c != b_to_c && a_to_c != a_to_b));

  const DimensionIndex a_rank = a_to_b->input_rank;
  const DimensionIndex b_rank = a_to_b->output_rank;
  const DimensionIndex c_rank = b_to_c->output_rank;

  a_to_c->input_rank = a_rank;
  a_to_c->output_rank = c_rank;

  CopyInputLabels(a_to_b, a_to_c,
                  /*can_move=*/can_move_from_a_to_b);

  BoxView<> b_to_c_domain = b_to_c->input_domain(b_rank);
  MutableBoxView<> a_to_c_domain = a_to_c->input_domain(a_rank);

  span<const OutputIndexMap> b_to_c_output_index_maps =
      b_to_c->output_index_maps().first(c_rank);
  span<const OutputIndexMap> a_to_b_output_index_maps =
      a_to_b->output_index_maps().first(b_rank);
  span<OutputIndexMap> a_to_c_output_index_maps =
      a_to_c->output_index_maps().first(c_rank);

  // Compute the input domain of the new `a_to_c` transform.
  TENSORSTORE_RETURN_IF_ERROR(
      PropagateBounds(b_to_c_domain, b_to_c->implicit_lower_bounds(b_rank),
                      b_to_c->implicit_upper_bounds(b_rank), a_to_b,
                      a_to_c_domain, a_to_c->implicit_lower_bounds(a_rank),
                      a_to_c->implicit_upper_bounds(a_rank)));

  // Compute the output index maps for each output dimension of the new `a_to_c`
  // transform.
  for (DimensionIndex c_dim = 0; c_dim < c_rank; ++c_dim) {
    auto& b_to_c_map = b_to_c_output_index_maps[c_dim];
    auto& a_to_c_map = a_to_c_output_index_maps[c_dim];
    const OutputIndexMethod b_to_c_method = b_to_c_map.stride() == 0
                                                ? OutputIndexMethod::constant
                                                : b_to_c_map.method();
    switch (b_to_c_method) {
      case OutputIndexMethod::constant: {
        a_to_c_map.SetConstant();
        a_to_c_map.stride() = 0;
        a_to_c_map.offset() = b_to_c_map.offset();
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex b_dim = b_to_c_map.input_dimension();
        ABSL_ASSERT(b_dim >= 0 && b_dim < b_rank);
        auto& a_to_b_map = a_to_b_output_index_maps[b_dim];
        const OutputIndexMethod a_to_b_method =
            a_to_b_map.stride() == 0 ? OutputIndexMethod::constant
                                     : a_to_b_map.method();
        // Compute the offset value of the new output index map.
        Index new_output_offset;
        if (internal::MulOverflow(a_to_b_map.offset(), b_to_c_map.stride(),
                                  &new_output_offset) ||
            internal::AddOverflow(b_to_c_map.offset(), new_output_offset,
                                  &a_to_c_map.offset())) {
          return absl::InvalidArgumentError(
              StrCat("Integer overflow computing output "
                     "offset for output dimension ",
                     c_dim, "."));
        }
        if (a_to_b_method == OutputIndexMethod::constant) {
          // Handle the single_input_dimension -> constant case.  Bounds were
          // already checked by PropagateBounds.
          a_to_c_map.SetConstant();
          a_to_c_map.stride() = 0;
          break;
        }
        // Compute the stride value of the new output index map.
        if (internal::MulOverflow(a_to_b_map.stride(), b_to_c_map.stride(),
                                  &a_to_c_map.stride())) {
          return absl::InvalidArgumentError(StrCat(
              "Integer overflow computing output_strides[", c_dim,
              "] = ", a_to_b_map.stride(), " * ", b_to_c_map.stride(), "."));
        }
        if (a_to_b_method == OutputIndexMethod::single_input_dimension) {
          // Handle the single_input_dimension -> single_input_dimension case.
          // Bounds were already checked by PropagateBounds.
          const DimensionIndex a_dim = a_to_b_map.input_dimension();
          ABSL_ASSERT(a_dim >= 0 && a_dim < a_rank);
          a_to_c_map.SetSingleInputDimension(a_dim);
          break;
        }
        // Handle the single_input_dimension -> array case.
        ABSL_ASSERT(a_to_b_method == OutputIndexMethod::array);
        const auto& a_to_b_index_array_data = a_to_b_map.index_array_data();
        // Compute the updated index_range bounds for this index array.
        IndexInterval index_range;
        {
          TENSORSTORE_ASSIGN_OR_RETURN(
              const IndexInterval propagated_bounds,
              GetAffineTransformDomain(
                  OptionallyImplicitIndexInterval{
                      b_to_c_domain[b_dim],
                      b_to_c->implicit_lower_bounds(b_rank)[b_dim],
                      b_to_c->implicit_upper_bounds(b_rank)[b_dim]}
                      .effective_interval(),
                  a_to_b_map.offset(), a_to_b_map.stride()));
          index_range =
              Intersect(a_to_b_index_array_data.index_range, propagated_bounds);
        }
        switch (GetArrayUniqueElementZeroOneManyCount(
            a_rank, a_to_c_domain.shape().data(),
            a_to_b_index_array_data.byte_strides)) {
          case ArrayUniqueElementZeroOneManyCount::kZero: {
            // Array has no elements.  Convert the index array to a constant
            // map.
            a_to_c_map.SetConstant();
            break;
          }
          case ArrayUniqueElementZeroOneManyCount::kOne: {
            // Convert index array map to constant map.
            a_to_c_map.SetConstant();
            TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
                a_to_b_index_array_data.element_pointer
                    .byte_strided_pointer()[IndexInnerProduct(
                        a_rank, a_to_b_index_array_data.byte_strides,
                        a_to_b->input_origin().data())],
                index_range, &a_to_c_map.offset(), &a_to_c_map.stride()));
            break;
          }
          case ArrayUniqueElementZeroOneManyCount::kMoreThanOne: {
            // TODO(jbms): move IndexArrayData if possible
            auto& index_array =
                a_to_c_map.SetArrayIndexing(a_rank, a_to_b_index_array_data);
            index_array.index_range = index_range;
            break;
          }
        }
        break;
      }
      case OutputIndexMethod::array: {
        // Handle array -> * case.  We simply rely on TransformArraySubRegion to
        // compute the new index array.
        auto& index_array_data = b_to_c_map.index_array_data();
        auto& a_to_c_map = a_to_c_output_index_maps[c_dim];
        auto& result_array_data = a_to_c_map.SetArrayIndexing(a_rank);
        result_array_data.index_range = index_array_data.index_range;
        auto transform_result = TransformArraySubRegion(
            SharedArrayView<const void, dynamic_rank, offset_origin>(
                index_array_data.element_pointer,
                StridedLayoutView<dynamic_rank, offset_origin>(
                    b_rank, b_to_c_domain.origin().data(),
                    b_to_c_domain.shape().data(),
                    index_array_data.byte_strides)),
            a_to_b, a_to_c_domain.origin().data(), a_to_c_domain.shape().data(),
            result_array_data.byte_strides,
            /*constraints=*/{});
        if (!transform_result) return transform_result.status();
        auto new_index_array_origin_pointer =
            StaticDataTypeCast<const Index, unchecked>(*transform_result);
        result_array_data.element_pointer = AddByteOffset(
            new_index_array_origin_pointer,
            -IndexInnerProduct(a_rank, result_array_data.byte_strides,
                               a_to_c_domain.origin().data()));
        Index output_offset = b_to_c_map.offset();
        Index output_stride = b_to_c_map.stride();
        switch (GetArrayUniqueElementZeroOneManyCount(
            a_rank, a_to_c_domain.shape().data(),
            result_array_data.byte_strides)) {
          case ArrayUniqueElementZeroOneManyCount::kZero:
            a_to_c_map.SetConstant();
            break;
          case ArrayUniqueElementZeroOneManyCount::kOne:
            // Convert index array map to constant map.
            TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
                *new_index_array_origin_pointer.data(),
                result_array_data.index_range, &output_offset, &output_stride));
            a_to_c_map.SetConstant();
            break;
          case ArrayUniqueElementZeroOneManyCount::kMoreThanOne:
            break;
        }
        a_to_c_map.offset() = output_offset;
        a_to_c_map.stride() = output_stride;
        break;
      }
    }
  }

  return absl::OkStatus();
}

Result<TransformRep::Ptr<>> ComposeTransforms(TransformRep* b_to_c,
                                              bool can_move_from_b_to_c,
                                              TransformRep* a_to_b,
                                              bool can_move_from_a_to_b) {
  ABSL_ASSERT(b_to_c);
  ABSL_ASSERT(a_to_b);
  const DimensionIndex a_rank = a_to_b->input_rank;
  const DimensionIndex b_rank = a_to_b->output_rank;
  const DimensionIndex c_rank = b_to_c->output_rank;
  if (b_rank != b_to_c->input_rank) {
    return absl::InvalidArgumentError(
        StrCat("Rank ", b_to_c->input_rank, " -> ", c_rank,
               " transform cannot be composed with rank ", a_rank, " -> ",
               b_rank, " transform."));
  }
  auto data = TransformRep::Allocate(a_rank, c_rank);
  TENSORSTORE_RETURN_IF_ERROR(ComposeTransforms(
      b_to_c, can_move_from_b_to_c, a_to_b, can_move_from_a_to_b, data.get()));
  return data;
}

}  // namespace internal_index_space
}  // namespace tensorstore
