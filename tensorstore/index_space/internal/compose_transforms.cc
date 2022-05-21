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

#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_array.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"

namespace tensorstore {
namespace internal_index_space {

namespace {

/// Returns `true` if `layout` specifies an index array map with a single
/// distinct element.
///
/// Note that zero-size dimensions with `byte_stride == 0` do not disqualify
/// `layout` from being considered a singleton map; it is assumed that these
/// dimensions have implicit bounds.
///
/// \param layout Array layout.
bool IsSingletonIndexArrayMap(StridedLayoutView<> layout) {
  for (DimensionIndex dim = 0, rank = layout.rank(); dim < rank; ++dim) {
    if (layout.byte_strides()[dim] == 0) continue;
    if (layout.shape()[dim] != 1) return false;
  }
  return true;
}

}  // namespace

absl::Status ComposeTransforms(TransformRep* b_to_c, bool can_move_from_b_to_c,
                               TransformRep* a_to_b, bool can_move_from_a_to_b,
                               TransformRep* a_to_c, bool domain_only) {
  assert(b_to_c != nullptr && a_to_b != nullptr && a_to_c != nullptr);
  const DimensionIndex a_to_c_output_rank =
      domain_only ? 0 : b_to_c->output_rank;
  assert(a_to_c_output_rank <= a_to_c->output_rank_capacity &&
         a_to_b->output_rank == b_to_c->input_rank &&
         a_to_b->input_rank <= a_to_c->input_rank_capacity);
  // Aliasing of `a_to_c` is not allowed
  assert(a_to_c != b_to_c && a_to_c != a_to_b);

  const DimensionIndex a_rank = a_to_b->input_rank;
  const DimensionIndex b_rank = a_to_b->output_rank;
  const DimensionIndex c_rank = b_to_c->output_rank;

  a_to_c->input_rank = a_rank;
  a_to_c->output_rank = a_to_c_output_rank;

  CopyInputLabels(a_to_b, a_to_c,
                  /*can_move=*/can_move_from_a_to_b);

  BoxView<> b_to_c_domain = b_to_c->input_domain(b_rank);
  MutableBoxView<> a_to_c_domain = a_to_c->input_domain(a_rank);

  // Compute the input domain of the new `a_to_c` transform.
  TENSORSTORE_RETURN_IF_ERROR(PropagateBounds(
      b_to_c_domain, b_to_c->implicit_lower_bounds,
      b_to_c->implicit_upper_bounds, a_to_b, a_to_c_domain,
      a_to_c->implicit_lower_bounds, a_to_c->implicit_upper_bounds));

  if (domain_only) {
    internal_index_space::DebugCheckInvariants(a_to_c);
    return absl::OkStatus();
  }

  span<const OutputIndexMap> b_to_c_output_index_maps =
      b_to_c->output_index_maps().first(c_rank);
  span<const OutputIndexMap> a_to_b_output_index_maps =
      a_to_b->output_index_maps().first(b_rank);
  span<OutputIndexMap> a_to_c_output_index_maps =
      a_to_c->output_index_maps().first(c_rank);

  const bool a_to_c_domain_is_explicitly_empty =
      IsDomainExplicitlyEmpty(a_to_c);

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
        assert(b_dim >= 0 && b_dim < b_rank);
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
          assert(a_dim >= 0 && a_dim < a_rank);
          a_to_c_map.SetSingleInputDimension(a_dim);
          break;
        }
        // Handle the single_input_dimension -> array case.
        assert(a_to_b_method == OutputIndexMethod::array);
        if (a_to_c_domain_is_explicitly_empty) {
          // Index array would contain zero elements, convert it to a constant
          // map.
          a_to_c_map.SetConstant();
          a_to_c_map.offset() = 0;
          a_to_c_map.stride() = 0;
          break;
        }
        const auto& a_to_b_index_array_data = a_to_b_map.index_array_data();
        // Compute the updated index_range bounds for this index array.
        IndexInterval index_range;
        {
          TENSORSTORE_ASSIGN_OR_RETURN(
              const IndexInterval propagated_bounds,
              GetAffineTransformDomain(
                  OptionallyImplicitIndexInterval{
                      b_to_c_domain[b_dim],
                      b_to_c->implicit_lower_bounds[b_dim],
                      b_to_c->implicit_upper_bounds[b_dim]}
                      .effective_interval(),
                  a_to_b_map.offset(), a_to_b_map.stride()));
          index_range =
              Intersect(a_to_b_index_array_data.index_range, propagated_bounds);
        }
        if (IsSingletonIndexArrayMap(
                StridedLayoutView<>(a_rank, a_to_c_domain.shape().data(),
                                    a_to_b_index_array_data.byte_strides))) {
          // Convert index array map to constant map.
          a_to_c_map.SetConstant();
          TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
              *a_to_b_index_array_data.array_view(a_to_b->input_domain(a_rank))
                   .byte_strided_origin_pointer(),
              index_range, &a_to_c_map.offset(), &a_to_c_map.stride()));
        } else {
          // TODO(jbms): move IndexArrayData if possible
          auto& index_array =
              a_to_c_map.SetArrayIndexing(a_rank, a_to_b_index_array_data);
          index_array.index_range = index_range;
        }
        break;
      }
      case OutputIndexMethod::array: {
        // Handle array -> * case.
        auto& a_to_c_map = a_to_c_output_index_maps[c_dim];
        if (a_to_c_domain_is_explicitly_empty) {
          // Index array would contain zero elements, convert it to a constant
          // map.
          a_to_c_map.SetConstant();
          a_to_c_map.offset() = 0;
          a_to_c_map.stride() = 0;
          break;
        }
        // Use TransformArraySubRegion to compute the new index array.
        auto& index_array_data = b_to_c_map.index_array_data();
        auto& result_array_data = a_to_c_map.SetArrayIndexing(a_rank);
        result_array_data.index_range = index_array_data.index_range;
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto transformed_element_pointer,
            TransformArraySubRegion(
                index_array_data.shared_array_view(b_to_c_domain), a_to_b,
                a_to_c_domain.origin().data(), a_to_c_domain.shape().data(),
                result_array_data.byte_strides,
                /*constraints=*/{skip_repeated_elements}));
        auto new_index_array_origin_pointer =
            StaticDataTypeCast<const Index, unchecked>(
                std::move(transformed_element_pointer));
        result_array_data.element_pointer = AddByteOffset(
            new_index_array_origin_pointer,
            -IndexInnerProduct(a_rank, result_array_data.byte_strides,
                               a_to_c_domain.origin().data()));
        Index output_offset = b_to_c_map.offset();
        Index output_stride = b_to_c_map.stride();
        if (IsSingletonIndexArrayMap(
                StridedLayoutView<>(a_rank, a_to_c_domain.shape().data(),
                                    result_array_data.byte_strides))) {
          // Convert index array map to constant map.
          TENSORSTORE_RETURN_IF_ERROR(ReplaceZeroRankIndexArrayIndexMap(
              *new_index_array_origin_pointer.data(),
              result_array_data.index_range, &output_offset, &output_stride));
          a_to_c_map.SetConstant();
        }
        a_to_c_map.offset() = output_offset;
        a_to_c_map.stride() = output_stride;
        break;
      }
    }
  }
  internal_index_space::DebugCheckInvariants(a_to_c);
  return absl::OkStatus();
}

Result<TransformRep::Ptr<>> ComposeTransforms(TransformRep* b_to_c,
                                              bool can_move_from_b_to_c,
                                              TransformRep* a_to_b,
                                              bool can_move_from_a_to_b,
                                              bool domain_only) {
  assert(b_to_c);
  assert(a_to_b);
  const DimensionIndex a_rank = a_to_b->input_rank;
  const DimensionIndex b_rank = a_to_b->output_rank;
  const DimensionIndex c_rank = b_to_c->output_rank;
  if (b_rank != b_to_c->input_rank) {
    return absl::InvalidArgumentError(
        StrCat("Rank ", b_to_c->input_rank, " -> ", c_rank,
               " transform cannot be composed with rank ", a_rank, " -> ",
               b_rank, " transform."));
  }
  auto data = TransformRep::Allocate(a_rank, domain_only ? 0 : c_rank);
  TENSORSTORE_RETURN_IF_ERROR(ComposeTransforms(b_to_c, can_move_from_b_to_c,
                                                a_to_b, can_move_from_a_to_b,
                                                data.get(), domain_only));
  return data;
}

Result<IndexTransform<dynamic_rank, dynamic_rank, container>> ComposeTransforms(
    IndexTransform<dynamic_rank, dynamic_rank, container> b_to_c,
    IndexTransform<dynamic_rank, dynamic_rank, container> a_to_b,
    bool domain_only) {
  auto b_to_c_rep = TransformAccess::rep(b_to_c);
  auto a_to_b_rep = TransformAccess::rep(a_to_b);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto a_to_c_rep,
      internal_index_space::ComposeTransforms(
          b_to_c_rep,
          /*can_move_from_b_to_c=*/b_to_c_rep->is_unique(), a_to_b_rep,
          /*can_move_from_a_to_b=*/a_to_b_rep->is_unique(), domain_only));
  return TransformAccess::Make<IndexTransform<>>(std::move(a_to_c_rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
