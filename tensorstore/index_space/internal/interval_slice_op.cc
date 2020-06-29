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

#include "tensorstore/index_space/internal/interval_slice_op.h"

#include "absl/container/fixed_array.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {
namespace {

struct InputDimensionIntervalSliceInfo {
  /// orig_input = new_input * stride + offset
  Index offset;

  /// stride value
  Index stride;
};

/// Validates the specified strided interval, and for each dimension in
/// `dimensions`, computes InputDimensionIntervalSliceInfo and updates
/// `input_origin` and `input_shape` to reflect the domain of the resultant
/// slice.
///
/// \param dimension_info[out] Array of length `input_rank` to which the slicing
///     information for each selected dimension will be stored.  The values for
///     dimensions not in `dimensions` will not be modified.  On error return,
///     may be partially modified.
/// \param transform[in,out] Specifies the transform for which the domain is to
///     be sliced.  On successful return, the domain is set to the domain of the
///     extracted slice..  On error return, may be partially modified.
/// \param dimensions[in] Specifies the selected dimensions to be sliced.
/// \param interval_form Specifies the interpretation of `stop_or_size_vector`.
/// \param translate If `true`, the origin of the selected dimensions are
///     translated to 0 in the new transform.
/// \param start_vector[in] Specifies the vector of start indices (of length
///     `dimensions->size()`) corresponding to the selected `dimensions`.
/// \param stop_or_size_vector[in] Specifies the vector of stop/size indices (of
///     length `dimensions->size()`).
/// \param stride_vector[in] Specifies the vector of strides (of length
///     `dimensions->size()`).
/// \returns `Status()` if the specified slice is valid.
/// \error `absl::StatusCode::kInvalidArgument` if the size of `start_vector`,
///     `stop_or_size_vector`, or `stride_vector` is not compatible with
///     `dimensions->size()`.
/// \error `absl::StatusCode::kInvalidArgument` if a stride value is `0` or
///     `std::numeric_limits<Index>::max()`.
/// \error `absl::StatusCode::kInvalidArgument` if `translate` is `true` but the
///     computed domain for one of the selected dimensions is unbounded below.
/// \error `absl::StatusCode::kOutOfRange` if the specified interval is invalid.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs while
///     computing the result.
static Status GetIntervalSliceInfo(
    span<InputDimensionIntervalSliceInfo> dimension_info,
    TransformRep* transform, span<const DimensionIndex> dimensions,
    IntervalForm interval_form, bool translate,
    IndexVectorOrScalar start_vector, IndexVectorOrScalar stop_or_size_vector,
    IndexVectorOrScalar stride_vector) {
  const DimensionIndex input_rank = dimension_info.size();
  ABSL_ASSERT(input_rank == transform->input_rank);
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    dimension_info[input_dim] = InputDimensionIntervalSliceInfo{0, 1};
  }
  auto compute_input_domain_slice = [&](DimensionIndex i,
                                        DimensionIndex input_dim) {
    const Index stride = stride_vector[i];
    const InputDimensionRef d = transform->input_dimension(input_dim);
    auto& info = dimension_info[input_dim];
    info.stride = stride;
    OptionallyImplicitIndexInterval new_domain;
    TENSORSTORE_RETURN_IF_ERROR(ComputeStridedSliceMap(
        d.optionally_implicit_domain(), interval_form,
        translate ? 0 : kImplicit, start_vector[i], stop_or_size_vector[i],
        stride, &new_domain, &info.offset));
    d.domain() = new_domain.interval();
    d.implicit_lower_bound() = new_domain.implicit_lower();
    d.implicit_upper_bound() = new_domain.implicit_upper();
    return absl::OkStatus();
  };
  for (DimensionIndex i = 0; i < dimensions.size(); ++i) {
    const DimensionIndex input_dim = dimensions[i];
    TENSORSTORE_RETURN_IF_ERROR(
        compute_input_domain_slice(i, input_dim),
        MaybeAnnotateStatus(
            _, StrCat("Computing interval slice for input dimension ",
                      input_dim)));
  }
  return absl::OkStatus();
}
}  // namespace

Result<IndexTransform<>> ApplyIntervalSliceOp(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    IntervalForm interval_form, bool translate,
    IndexVectorOrScalar start_vector, IndexVectorOrScalar stop_or_size_vector,
    IndexVectorOrScalar stride_vector) {
  const DimensionIndex num_dims = dimensions->size();
  const DimensionIndex input_rank = transform.input_rank();
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(start_vector, num_dims));
  TENSORSTORE_RETURN_IF_ERROR(
      CheckIndexVectorSize(stop_or_size_vector, num_dims));
  TENSORSTORE_RETURN_IF_ERROR(CheckIndexVectorSize(stride_vector, num_dims));
  TransformRep::Ptr<> rep =
      MutableRep(TransformAccess::rep_ptr<container>(std::move(transform)));
  absl::FixedArray<InputDimensionIntervalSliceInfo, internal::kNumInlinedDims>
      input_dimension_info(input_rank);
  // Computes slicing parameters and updates `input_origin` and `input_shape`.
  TENSORSTORE_RETURN_IF_ERROR(GetIntervalSliceInfo(
      input_dimension_info, rep.get(), *dimensions, interval_form, translate,
      start_vector, stop_or_size_vector, stride_vector));
  const DimensionIndex output_rank = rep->output_rank;
  span<OutputIndexMap> maps = rep->output_index_maps().first(output_rank);
  // Updates the output index maps.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = maps[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        const auto& slice_info = input_dimension_info[input_dim];
        Index offset;
        if (internal::MulOverflow(slice_info.offset, map.stride(), &offset) ||
            internal::AddOverflow(offset, map.offset(), &map.offset())) {
          return absl::InvalidArgumentError(
              StrCat("Integer overflow computing offset for output dimension ",
                     output_dim));
        }
        if (internal::MulOverflow(slice_info.stride, map.stride(),
                                  &map.stride())) {
          return absl::InvalidArgumentError(
              StrCat("Integer overflow computing stride for output dimension ",
                     output_dim));
        }
        break;
      }
      case OutputIndexMethod::array: {
        auto& index_array_data = map.index_array_data();
        Index element_pointer_byte_offset = 0;
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          const auto& slice_info = input_dimension_info[input_dim];
          Index& byte_stride = index_array_data.byte_strides[input_dim];
          element_pointer_byte_offset = internal::wrap_on_overflow::Add(
              element_pointer_byte_offset, internal::wrap_on_overflow::Multiply(
                                               byte_stride, slice_info.offset));
          byte_stride = internal::wrap_on_overflow::Multiply(byte_stride,
                                                             slice_info.stride);
        }
        index_array_data.element_pointer =
            AddByteOffset(std::move(index_array_data.element_pointer),
                          element_pointer_byte_offset);
        break;
      }
    }
  }
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

}  // namespace internal_index_space
}  // namespace tensorstore
