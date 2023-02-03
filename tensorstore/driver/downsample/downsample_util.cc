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

#include "tensorstore/driver/downsample/downsample_util.h"

#include <algorithm>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/identity_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

std::ostream& operator<<(std::ostream& os,
                         const PropagatedIndexTransformDownsampling& x) {
  return os << "transform=" << x.transform << "\ninput_downsample_factors="
            << absl::StrJoin(x.input_downsample_factors, ",");
}

namespace {
/// Computes the number of output index maps that depend on each input dimension
/// of `transform`.
///
/// This is used by `PropagateIndexTransformDownsampling`.
absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
ComputeInputDimensionReferenceCounts(IndexTransformView<> transform) {
  using internal_index_space::TransformAccess;

  assert(transform.valid());
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      input_dimension_ref_counts(input_rank, false);
  auto transform_rep = TransformAccess::rep(transform);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& output_map = transform_rep->output_index_maps()[output_dim];
    switch (output_map.method()) {
      case OutputIndexMethod::constant:
        break;
      case OutputIndexMethod::single_input_dimension:
        ++input_dimension_ref_counts[output_map.input_dimension()];
        break;
      case OutputIndexMethod::array: {
        const auto& index_array_data = output_map.index_array_data();
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          if (index_array_data.byte_strides[input_dim] != 0) {
            ++input_dimension_ref_counts[input_dim];
          }
        }
        break;
      }
    }
  }
  return input_dimension_ref_counts;
}

/// Determines the number of additional input dimensions needed by
/// `PropagateIndexTransformDownsampling`.
///
/// \param downsampled_transform Transform to the downsampled output domain.
/// \param output_downsample_factors Downsample factors for each output
///     dimension.
/// \param input_dimension_ref_counts Must equal result of
///     `ComputeInputDimensionReferenceCounts`.
/// \param is_domain_empty Must equal
///     `downsampled_transform.domain().box().is_empty()`.
/// \returns The number of additional input dimensions.
DimensionIndex ComputeAdditionalInputDimensionsNeeded(
    IndexTransformView<> downsampled_transform,
    span<const Index> output_downsample_factors,
    span<DimensionIndex> input_dimension_ref_counts, bool is_domain_empty) {
  using internal_index_space::TransformAccess;

  assert(downsampled_transform.valid());
  const DimensionIndex output_rank = downsampled_transform.output_rank();
  assert(input_dimension_ref_counts.size() ==
         downsampled_transform.input_rank());
  assert(output_downsample_factors.size() == output_rank);
  DimensionIndex additional_input_dims = 0;
  auto old_transform_rep = TransformAccess::rep(downsampled_transform);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    assert(output_downsample_factors[output_dim] > 0);
    if (output_downsample_factors[output_dim] == 1) {
      // Output dimension is not strided, need not change.
      continue;
    }
    const auto& output_map = old_transform_rep->output_index_maps()[output_dim];
    switch (output_map.method()) {
      case OutputIndexMethod::constant:
        // Will be converted to `single_input_dimension`.
        if (!is_domain_empty) {
          ++additional_input_dims;
        }
        break;
      case OutputIndexMethod::single_input_dimension:
        // If a given strided output dimension depends on exactly one input
        // dimension with stride 1 or -1, then we can just adjust it.
        //
        // Otherwise, we need to use an index array and a new dimension.
        if ((std::abs(output_map.stride()) != 1 ||
             input_dimension_ref_counts[output_map.input_dimension()] != 1) &&
            !downsampled_transform.input_domain()
                 .box()[output_map.input_dimension()]
                 .empty()) {
          ++additional_input_dims;
        }
        break;
      case OutputIndexMethod::array: {
        ++additional_input_dims;
        break;
      }
    }
  }
  return additional_input_dims;
}

/// Extends `output_map` to `new_input_rank`, and assigns to `new_output_map`.
///
/// This is used by `PropagateIndexTransformDownsampling`.
///
/// \param output_map Existing output map.
/// \param new_output_map[out] New output map to assign.
/// \param input_rank Input rank of `output_map`.
/// \param new_input_rank Input rank of `new_output_map` (only relevant for
///     `OutputIndexMethod::array`).
absl::Status ExtendOutputIndexMap(
    const internal_index_space::OutputIndexMap& output_map,
    internal_index_space::OutputIndexMap& new_output_map,
    DimensionIndex input_rank, DimensionIndex new_input_rank) {
  new_output_map.offset() = output_map.offset();
  // Unconditionally copy stride.  While not strictly necessary for
  // `OutputIndexMethod::constant`, leaving the stride uninitialized in that
  // case leads to MemorySanitizer use-of-uninitialized-value errors because
  // some code paths check for `stride == 0` before checking if the method is
  // `constant`.
  new_output_map.stride() = output_map.stride();
  switch (output_map.method()) {
    case OutputIndexMethod::constant:
      new_output_map.SetConstant();
      break;
    case OutputIndexMethod::single_input_dimension:
      new_output_map.SetSingleInputDimension(output_map.input_dimension());
      break;
    case OutputIndexMethod::array: {
      const auto& index_array_data = output_map.index_array_data();
      auto& new_index_array_data =
          new_output_map.SetArrayIndexing(new_input_rank);
      new_index_array_data.element_pointer = index_array_data.element_pointer;
      new_index_array_data.index_range = index_array_data.index_range;
      std::copy_n(index_array_data.byte_strides, input_rank,
                  new_index_array_data.byte_strides);
      std::fill_n(new_index_array_data.byte_strides + input_rank,
                  new_input_rank - input_rank, Index(0));
      break;
    }
  }
  return absl::OkStatus();
}

/// Propagates a downsampling factor through a `single_input_dimension` output
/// index map with stride of +/-1.
///
/// This is used by `PropagateIndexTransformDownsampling`.
///
/// \param original_offset Offset of the original output index map.
/// \param original_stride Stride of the original output index map.  Must be `1`
///     or `-1`.
/// \param input_interval Bounds for the corresponding input dimension in the
///     downsampled input domain.
/// \param output_downsample_factor Downsample factor to propagate.
/// \param new_output_map[out] New output index map to assign.
/// \param output_base_bounds Bounds for this dimension in the base output
///     space.
/// \param new_input_domain[out] Input domain of `propagated.transform` to
///     update for `new_input_dim`.
/// \param new_input_dim New input dimension index.
/// \param propagated[out] Propagated transform.
absl::Status PropagateUnitStrideSingleInputDimensionMapDownsampling(
    Index original_offset, Index original_stride, IndexInterval input_interval,
    Index output_downsample_factor,
    internal_index_space::OutputIndexMap& new_output_map,
    IndexInterval output_base_bounds, MutableBoxView<> new_input_domain,
    DimensionIndex new_input_dim,
    PropagatedIndexTransformDownsampling& propagated) {
  assert(original_stride == 1 || original_stride == -1);
  if (internal::MulOverflow(original_offset, output_downsample_factor,
                            &new_output_map.offset())) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Integer overflow computing output offset ",
                            original_offset, " * ", output_downsample_factor));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto bounds_interval,
      GetAffineTransformDomain(output_base_bounds, new_output_map.offset(),
                               original_stride));
  auto input_bounds = DownsampleInterval(
      bounds_interval, output_downsample_factor, DownsampleMethod::kMean);
  if (!Contains(input_bounds, input_interval)) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Propagated bounds interval ", input_bounds,
                            " does not contain ", input_interval));
  }
  propagated.input_downsample_factors[new_input_dim] = output_downsample_factor;
  new_output_map.SetSingleInputDimension(new_input_dim);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_interval,
      GetAffineTransformInverseDomain(
          input_interval, 0, original_stride * output_downsample_factor));
  new_interval = Intersect(new_interval, bounds_interval);
  new_output_map.stride() = original_stride;
  new_input_domain[new_input_dim] = new_interval;
  return absl::OkStatus();
}

/// Propagates a downsampling factor through a `single_input_dimension` output
/// index map with non-unit stride or an input dimension that is also referenced
/// from other output index maps.
///
/// In this case a new input dimension is always added.
///
/// This is used by `PropagateIndexTransformDownsampling`.
///
/// \param output_map[in] Original output index map.
/// \pre `output_map.method() == OutputIndexMethod::single_input_dimension`
/// \param input_interval Bounds for the corresponding input dimension in the
///     downsampled input domain.
/// \param output_downsample_factor Downsample factor to propagate.
/// \param new_output_map[out] New output index map to assign.
/// \param output_base_bounds Bounds for this dimension in the base output
///     space.
/// \param new_input_domain[out] Input domain of `propagated.transform` to
///     update for `new_input_dim`.
/// \param new_input_dim New input dimension index.
/// \param propagated[out] Propagated transform.
absl::Status PropagateSingleInputDimensionMapDownsamplingAsNewDimension(
    const internal_index_space::OutputIndexMap& output_map,
    IndexInterval input_interval, Index output_downsample_factor,
    internal_index_space::OutputIndexMap& new_output_map,
    IndexInterval output_base_bounds, MutableBoxView<> new_input_domain,
    DimensionIndex new_input_dim,
    PropagatedIndexTransformDownsampling& propagated) {
  if (input_interval.size() == 1 || output_map.stride() == 0) {
    // Convert to constant map.  This avoids allocating an index array
    // and makes the bounds checking logic below simpler since it does
    // not have to handle this case.
    Index adjusted_offset;
    if (internal::MulOverflow(input_interval.inclusive_min(),
                              output_map.stride(), &adjusted_offset) ||
        internal::AddOverflow(adjusted_offset, output_map.offset(),
                              &adjusted_offset)) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Integer overflow computing offset ", output_map.offset(), " + ",
          input_interval.inclusive_min(), " * ", output_map.stride()));
    }
    return PropagateUnitStrideSingleInputDimensionMapDownsampling(
        /*original_offset=*/adjusted_offset, /*original_stride=*/1,
        /*input_interval=*/IndexInterval::UncheckedSized(0, 1),
        output_downsample_factor, new_output_map, output_base_bounds,
        new_input_domain, new_input_dim, propagated);
  }
  propagated.input_downsample_factors[new_input_dim] = output_downsample_factor;
  if (output_downsample_factor > kInfIndex) {
    return absl::OutOfRangeError("Downsample factor is out of range");
  }
  new_input_domain[new_input_dim] =
      IndexInterval::UncheckedSized(0, output_downsample_factor);
  // Convert to index array map.
  new_output_map.offset() = 0;
  new_output_map.stride() = 1;
  auto& new_index_array_data =
      new_output_map.SetArrayIndexing(new_input_domain.rank());
  new_index_array_data.index_range = output_base_bounds;
  Index adjusted_stride;
  Index adjusted_offset;
  if (internal::MulOverflow(output_map.stride(), output_downsample_factor,
                            &adjusted_stride)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow computing stride ", output_map.stride(), " * ",
        output_downsample_factor));
  }
  if (internal::MulOverflow(output_map.offset(), output_downsample_factor,
                            &adjusted_offset)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow computing offset ", output_map.offset(), " * ",
        output_downsample_factor));
  }
  if (!input_interval.empty()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto output_range,
        GetAffineTransformRange(input_interval, adjusted_offset,
                                adjusted_stride));
    TENSORSTORE_ASSIGN_OR_RETURN(
        output_range,
        ShiftInterval(output_range, output_downsample_factor - 1, 0));
    if (!Contains(output_base_bounds, output_range)) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Output bounds interval ", output_base_bounds,
          " does not contain output range interval ", output_range));
    }
  }
  // Use `new_index_array_data.byte_strides` as temporary buffer for calling
  // `AllocateArrayElementsLike`.  Copy the byte strides of the existing index
  // array, and add an additional dimension
  std::fill_n(new_index_array_data.byte_strides, new_input_domain.rank(),
              Index(0));
  new_index_array_data.byte_strides[output_map.input_dimension()] = 1;
  new_index_array_data.byte_strides[new_input_dim] = 2;
  new_index_array_data.element_pointer = AllocateArrayElementsLike<Index>(
      new_index_array_data.layout(new_input_domain),
      new_index_array_data.byte_strides, skip_repeated_elements);
  Index* array_origin =
      const_cast<Index*>(new_index_array_data.array_view(new_input_domain)
                             .byte_strided_origin_pointer()
                             .get());
  for (Index j = 0; j < input_interval.size(); ++j) {
    const Index base_index =
        adjusted_offset +
        adjusted_stride * (input_interval.inclusive_min() + j);
    for (Index i = 0; i < output_downsample_factor; ++i) {
      Index x;
      if (internal::AddOverflow(base_index, i, &x) ||
          x > output_base_bounds.inclusive_max()) {
        x = output_base_bounds.inclusive_max();
      } else if (x < output_base_bounds.inclusive_min()) {
        x = output_base_bounds.inclusive_min();
      }
      array_origin[input_interval.size() * i + j] = x;
    }
  }
  return absl::OkStatus();
}

/// Creates a constant output map corresponding to an output index map that
/// requires a new input dimension in the propagated transform.
///
/// This is used to create a dummy constant output map in the case where the
/// downsampled transform has an empty domain.
///
/// \param output_downsample_factor Downsample factor to propagate.
/// \param new_output_map[out] New output index map to assign.
/// \param new_input_domain[in,out] Input domain of `propagated.transform` to
///     update for `new_input_dim`.
/// \param new_input_dim New input dimension index.
/// \param propagated[out] Propagated transform.
absl::Status PropagateIndexMapThatRequiresNewInputDimensionForEmptyDomain(
    Index output_downsample_factor,
    internal_index_space::OutputIndexMap& new_output_map,
    MutableBoxView<> new_input_domain, DimensionIndex new_input_dim,
    PropagatedIndexTransformDownsampling& propagated) {
  propagated.input_downsample_factors[new_input_dim] = output_downsample_factor;
  if (output_downsample_factor > kInfIndex) {
    return absl::OutOfRangeError("Downsample factor is out of range");
  }
  new_input_domain[new_input_dim] =
      IndexInterval::UncheckedSized(0, output_downsample_factor);
  new_output_map.SetConstant();
  new_output_map.offset() = 0;
  new_output_map.stride() = 0;
  return absl::OkStatus();
}

/// Propagates a downsampling factor through an index array output index map.
///
/// \param output_map Original output index map.
/// \pre `output_map.method() == OutputIndexMethod::array`
/// \param downsampled_input_domain Input domain of original transform.
/// \param output_downsample_factor Downsample factor to propagate.
/// \param new_output_map[out] New output index map to assign.
/// \param output_base_bounds Bounds for this dimension in the base output
///     space.
/// \param new_input_domain[in,out] Input domain of `propagated.transform` to
///     update for `new_input_dim`.
/// \param new_input_dim New input dimension index.
/// \param propagated[out] Propagated transform.
absl::Status PropagateIndexArrayMapDownsampling(
    const internal_index_space::OutputIndexMap& output_map,
    BoxView<> downsampled_input_domain, Index output_downsample_factor,
    internal_index_space::OutputIndexMap& new_output_map,
    IndexInterval output_base_bounds, MutableBoxView<> new_input_domain,
    DimensionIndex new_input_dim,
    PropagatedIndexTransformDownsampling& propagated) {
  new_output_map.offset() = 0;
  propagated.input_downsample_factors[new_input_dim] = output_downsample_factor;
  if (output_downsample_factor > kInfIndex) {
    return absl::OutOfRangeError("Downsample factor is out of range");
  }
  new_input_domain[new_input_dim] =
      IndexInterval::UncheckedSized(0, output_downsample_factor);
  const DimensionIndex input_rank = downsampled_input_domain.rank();
  const auto& index_array_data = output_map.index_array_data();
  new_output_map.stride() = 1;
  auto& new_index_array_data =
      new_output_map.SetArrayIndexing(new_input_domain.rank());
  Index adjusted_stride;
  Index adjusted_offset;
  if (internal::MulOverflow(output_map.stride(), output_downsample_factor,
                            &adjusted_stride)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow computing stride ", output_map.stride(), " * ",
        output_downsample_factor));
  }
  if (internal::MulOverflow(output_map.offset(), output_downsample_factor,
                            &adjusted_offset)) {
    return absl::OutOfRangeError(tensorstore::StrCat(
        "Integer overflow computing offset ", output_map.offset(), " * ",
        output_downsample_factor));
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto padded_output_interval,
      ShiftInterval(output_base_bounds, -(output_downsample_factor - 1), 0));

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto effective_index_range,
      GetAffineTransformDomain(padded_output_interval, adjusted_offset,
                               adjusted_stride));
  effective_index_range =
      Intersect(effective_index_range, index_array_data.index_range);
  new_index_array_data.index_range = output_base_bounds;
  // Use `new_index_array_data.byte_strides` as temporary buffer for
  // calling `AllocateArrayLike`.  Copy the byte strides of the existing
  // index array, and add an additional dimension
  std::copy_n(index_array_data.byte_strides, input_rank,
              new_index_array_data.byte_strides);
  std::fill_n(new_index_array_data.byte_strides + input_rank,
              new_input_domain.rank() - input_rank, Index(0));
  new_index_array_data.byte_strides[new_input_dim] =
      std::numeric_limits<Index>::max();
  // Note that we pass `new_input_domain` to `AllocateArrayElementsLike` even
  // though it is only partially initialized; only dimensions up to
  // `new_input_dim` are initialized.  The dimensions after `new_input_dim` have
  // the origin set to 0 and the shape set to 1, to ensure they are "skipped"
  // (i.e. byte_stride set to 0).
  new_index_array_data.element_pointer = AllocateArrayElementsLike<Index>(
      new_index_array_data.layout(new_input_domain),
      new_index_array_data.byte_strides, skip_repeated_elements);
  absl::Status status;
  IterateOverArrays(
      [&](const Index* existing_index,
          ByteStridedPointer<const Index> new_index) {
        const Index existing_index_value = *existing_index;
        if (!Contains(effective_index_range, existing_index_value)) {
          status = CheckContains(effective_index_range, existing_index_value);
          return false;
        }
        Index base_index =
            existing_index_value * adjusted_stride + adjusted_offset;
        const Index byte_stride =
            new_index_array_data.byte_strides[new_input_dim];
        Index cur_index =
            std::max(base_index, output_base_bounds.inclusive_min());
        for (Index i = 0; i < output_downsample_factor; ++i) {
          Index x;
          if (!internal::AddOverflow(base_index, i, &x) &&
              output_base_bounds.exclusive_max() > x) {
            cur_index = std::max(cur_index, x);
          }
          assert(Contains(output_base_bounds, cur_index));
          *const_cast<Index*>((new_index + i * byte_stride).get()) = cur_index;
        }
        return true;
      },
      skip_repeated_elements,
      index_array_data.array_view(downsampled_input_domain),
      new_index_array_data.array_view(downsampled_input_domain));
  return status;
}

}  // namespace

absl::Status PropagateIndexTransformDownsampling(
    IndexTransformView<> downsampled_transform, BoxView<> output_base_bounds,
    span<const Index> output_downsample_factors,
    PropagatedIndexTransformDownsampling& propagated) {
  using internal_index_space::TransformAccess;
  using internal_index_space::TransformRep;

  assert(downsampled_transform.valid());
  const DimensionIndex output_rank = downsampled_transform.output_rank();
  const DimensionIndex input_rank = downsampled_transform.input_rank();
  assert(output_base_bounds.rank() == output_rank);
  assert(output_downsample_factors.size() == output_rank);

  auto input_dimension_ref_counts =
      ComputeInputDimensionReferenceCounts(downsampled_transform);
  const bool is_domain_empty = downsampled_transform.domain().box().is_empty();
  DimensionIndex additional_input_dims = ComputeAdditionalInputDimensionsNeeded(
      downsampled_transform, output_downsample_factors,
      input_dimension_ref_counts, is_domain_empty);

  const DimensionIndex new_input_rank = input_rank + additional_input_dims;
  TENSORSTORE_RETURN_IF_ERROR(ValidateRank(new_input_rank));
  auto new_transform = TransformRep::Allocate(new_input_rank, output_rank);
  new_transform->output_rank = output_rank;
  internal_index_space::CopyTransformRepDomain(
      TransformAccess::rep(downsampled_transform), new_transform.get());
  new_transform->input_rank = new_input_rank;
  // Bounds of the new transform are always marked explicit, because they are
  // computed specifically for the exact output bounds, and the normal
  // propagation used for implicit bounds would not do the right thing.  It is
  // assumed that any implicit bounds have already been resolved.
  new_transform->implicit_lower_bounds = false;
  new_transform->implicit_upper_bounds = false;
  MutableBoxView<> input_domain = new_transform->input_domain(new_input_rank);
  // Initialize origin and shape for new input dimensions.  The origin does not
  // matter, but is set to avoid a spurious MemorySanitizer
  // use-of-uninitialized-value error in PropagateIndexArrayMapDownsampling.
  // The shape is set to 1 to ensure uninitialized dimensions are skipped when
  // allocating index arrays.
  std::fill(input_domain.origin().begin() + input_rank,
            input_domain.origin().begin() + new_input_rank, Index(0));
  std::fill(input_domain.shape().begin() + input_rank,
            input_domain.shape().begin() + new_input_rank, Index(1));
  propagated.input_downsample_factors.clear();
  propagated.input_downsample_factors.resize(new_input_rank, 1);
  DimensionIndex next_input_dim = input_rank;
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& output_map = TransformAccess::rep(downsampled_transform)
                                 ->output_index_maps()[output_dim];
    auto& new_output_map = new_transform->output_index_maps()[output_dim];
    const Index output_downsample_factor =
        output_downsample_factors[output_dim];
    if (output_downsample_factor == 1) {
      // No downsampling of this output dimension, just copy existing output
      // map.
      TENSORSTORE_RETURN_IF_ERROR(ExtendOutputIndexMap(
          output_map, new_output_map, input_rank, new_input_rank));
      continue;
    }
    absl::Status status;
    switch (output_map.method()) {
      case OutputIndexMethod::constant: {
        if (is_domain_empty) {
          new_output_map.SetConstant();
          new_output_map.offset() = 0;
          new_output_map.stride() = 0;
          break;
        }
        // New input dimension is required.
        status = PropagateUnitStrideSingleInputDimensionMapDownsampling(
            /*original_offset=*/output_map.offset(), /*original_stride=*/1,
            /*input_interval=*/IndexInterval::UncheckedSized(0, 1),
            output_downsample_factor, new_output_map,
            /*output_base_bounds=*/output_base_bounds[output_dim], input_domain,
            /*new_input_dim=*/next_input_dim++, propagated);
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = output_map.input_dimension();
        const IndexInterval input_interval =
            downsampled_transform.input_domain().box()[input_dim];
        // Check if this map can be handled without creating a new input
        // dimension.
        if (std::abs(output_map.stride()) == 1 &&
            input_dimension_ref_counts[input_dim] == 1) {
          status = PropagateUnitStrideSingleInputDimensionMapDownsampling(
              /*original_offset=*/output_map.offset(),
              /*original_stride=*/output_map.stride(),
              /*input_interval=*/input_interval, output_downsample_factor,
              new_output_map,
              /*output_base_bounds=*/output_base_bounds[output_dim],
              input_domain,
              /*new_input_dim=*/input_dim, propagated);
          break;
        }
        if (!IsFinite(input_interval)) {
          status = absl::InvalidArgumentError(tensorstore::StrCat(
              "Input domain ", input_interval, " is not finite"));
          break;
        }
        if (input_interval.empty()) {
          new_output_map.SetSingleInputDimension(input_dim);
          new_output_map.offset() = 0;
          new_output_map.stride() = 1;
          break;
        }
        // New input dimension is required.
        status =
            is_domain_empty
                ? PropagateIndexMapThatRequiresNewInputDimensionForEmptyDomain(
                      output_downsample_factor, new_output_map, input_domain,
                      next_input_dim++, propagated)
                : PropagateSingleInputDimensionMapDownsamplingAsNewDimension(
                      output_map, input_interval, output_downsample_factor,
                      new_output_map, output_base_bounds[output_dim],
                      input_domain, next_input_dim++, propagated);
        break;
      }
      case OutputIndexMethod::array: {
        status =
            is_domain_empty
                ? PropagateIndexMapThatRequiresNewInputDimensionForEmptyDomain(
                      output_downsample_factor, new_output_map, input_domain,
                      next_input_dim++, propagated)
                : PropagateIndexArrayMapDownsampling(
                      output_map, downsampled_transform.domain().box(),
                      output_downsample_factor, new_output_map,
                      output_base_bounds[output_dim], input_domain,
                      next_input_dim++, propagated);
        break;
      }
    }
    if (!status.ok()) {
      return tensorstore::MaybeAnnotateStatus(
          status,
          tensorstore::StrCat("Propagating downsampling factor ",
                              output_downsample_factor,
                              " through output dimension ", output_dim));
    }
  }
  internal_index_space::DebugCheckInvariants(new_transform.get());
  propagated.transform =
      internal_index_space::TransformAccess::Make<IndexTransform<>>(
          std::move(new_transform));
  return absl::OkStatus();
}

Result<PropagatedIndexTransformDownsampling>
PropagateIndexTransformDownsampling(
    IndexTransformView<> downsampled_transform, BoxView<> output_base_bounds,
    span<const Index> output_downsample_factors) {
  PropagatedIndexTransformDownsampling propagated;
  TENSORSTORE_RETURN_IF_ERROR(PropagateIndexTransformDownsampling(
      downsampled_transform, output_base_bounds, output_downsample_factors,
      propagated));
  return propagated;
}

IndexInterval DownsampleInterval(IndexInterval base_interval,
                                 Index downsample_factor,
                                 DownsampleMethod method) {
  assert(downsample_factor > 0);
  Index inclusive_min;
  if (base_interval.inclusive_min() == -kInfIndex) {
    inclusive_min = -kInfIndex;
  } else {
    switch (method) {
      case DownsampleMethod::kStride:
        // Round up to ensure `base_interval` contains
        // `downsampled_inclusive_min * downsample_factor`.
        inclusive_min =
            CeilOfRatio(base_interval.inclusive_min(), downsample_factor);
        break;
      case DownsampleMethod::kMean:
      case DownsampleMethod::kMin:
      case DownsampleMethod::kMax:
      case DownsampleMethod::kMedian:
      case DownsampleMethod::kMode:
        // Round down since region-based methods can be computed from just a
        // single element.
        inclusive_min =
            FloorOfRatio(base_interval.inclusive_min(), downsample_factor);
        break;
      default:
        ABSL_UNREACHABLE();  // COV_NF_LINE
    }
  }
  Index inclusive_max;
  if (base_interval.inclusive_max() == kInfIndex) {
    inclusive_max = kInfIndex;
  } else if (base_interval.empty()) {
    inclusive_max = inclusive_min - 1;
  } else {
    // Round down for all downsampling methods.
    inclusive_max =
        FloorOfRatio(base_interval.inclusive_max(), downsample_factor);
  }
  return IndexInterval::UncheckedClosed(inclusive_min, inclusive_max);
}

void DownsampleBounds(BoxView<> base_bounds,
                      MutableBoxView<> downsampled_bounds,
                      span<const Index> downsample_factors,
                      DownsampleMethod method) {
  const DimensionIndex rank = base_bounds.rank();
  assert(rank == downsampled_bounds.rank());
  assert(rank == downsample_factors.size());
  for (DimensionIndex i = 0; i < rank; ++i) {
    downsampled_bounds[i] =
        DownsampleInterval(base_bounds[i], downsample_factors[i], method);
  }
}

namespace {
class DownsampleDomainBuilder {
 public:
  explicit DownsampleDomainBuilder(IndexDomainView<> base_domain,
                                   bool domain_only) {
    const DimensionIndex input_rank = base_domain.rank();
    const DimensionIndex output_rank = domain_only ? 0 : input_rank;
    rep = internal_index_space::TransformRep::Allocate(input_rank, output_rank);
    rep->input_rank = input_rank;
    rep->output_rank = output_rank;
    rep->implicit_lower_bounds = base_domain.implicit_lower_bounds();
    rep->implicit_upper_bounds = base_domain.implicit_upper_bounds();
    const auto& labels = base_domain.labels();
    std::copy(labels.begin(), labels.end(), rep->input_labels().begin());
    if (!domain_only) {
      internal_index_space::SetToIdentityTransform(rep->output_index_maps());
    }
  }

  MutableBoxView<> InputBounds() { return rep->input_domain(rep->input_rank); }

  IndexTransform<> MakeTransform() {
    internal_index_space::DebugCheckInvariants(rep.get());
    return internal_index_space::TransformAccess::Make<IndexTransform<>>(
        std::move(rep));
  }

 private:
  internal_index_space::TransformRep::Ptr<> rep;
};
}  // namespace

IndexDomain<> DownsampleDomain(IndexDomainView<> base_domain,
                               span<const Index> downsample_factors,
                               DownsampleMethod method) {
  DownsampleDomainBuilder builder(base_domain, /*domain_only=*/true);
  DownsampleBounds(base_domain.box(), builder.InputBounds(), downsample_factors,
                   method);
  return builder.MakeTransform().domain();
}

IndexTransform<> GetDownsampledDomainIdentityTransform(
    IndexDomainView<> base_domain, span<const Index> downsample_factors,
    DownsampleMethod method) {
  DownsampleDomainBuilder builder(base_domain, /*domain_only=*/false);
  DownsampleBounds(base_domain.box(), builder.InputBounds(), downsample_factors,
                   method);
  return builder.MakeTransform();
}

bool CanDownsampleIndexTransform(IndexTransformView<> base_transform,
                                 BoxView<> base_bounds,
                                 span<const Index> downsample_factors) {
  const Index output_rank = base_transform.output_rank();
  assert(base_bounds.rank() == output_rank);
  assert(downsample_factors.size() == output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const Index downsample_factor = downsample_factors[output_dim];
    const auto base_interval = base_bounds[output_dim];
    const auto map = base_transform.output_index_maps()[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant:
        if (downsample_factor != 1 &&
            ((base_interval.inclusive_min() != map.offset() &&
              ((map.offset() % downsample_factor) != 0)) ||
             (base_interval.inclusive_max() != map.offset() &&
              ((map.offset() + 1) % downsample_factor) != 0))) {
          // single index is not aligned to a downsampling block
          return false;
        }
        break;
      case OutputIndexMethod::single_input_dimension: {
        if (downsample_factor == 1) break;
        if (map.stride() != 1 && map.stride() != -1) {
          return false;
        }
        const auto input_interval =
            base_transform.input_domain().box()[map.input_dimension()];
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto shifted_interval,
            GetAffineTransformRange(input_interval, map.offset(), map.stride()),
            false);
        if ((base_interval.inclusive_min() !=
                 shifted_interval.inclusive_min() &&
             (shifted_interval.inclusive_min() % downsample_factor) != 0) ||
            (base_interval.exclusive_max() !=
                 shifted_interval.exclusive_max() &&
             (shifted_interval.exclusive_max() % downsample_factor) != 0)) {
          return false;
        }
        break;
      }
      case OutputIndexMethod::array:
        // Chunks with index array maps cannot be downsampled independently,
        // since our tracking of chunks does not handle non-rectangular regions.
        return false;
    }
  }
  return true;
}

}  // namespace internal_downsample
}  // namespace tensorstore
