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

#include "tensorstore/index_space/index_transform.h"

#include <numeric>

#include "absl/container/fixed_array.h"
#include "tensorstore/box.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

namespace internal_index_space {
std::string DescribeTransformForCast(DimensionIndex input_rank,
                                     DimensionIndex output_rank) {
  return StrCat("index transform with input ",
                StaticCastTraits<DimensionIndex>::Describe(input_rank),
                " and output ",
                StaticCastTraits<DimensionIndex>::Describe(output_rank));
}
std::string DescribeDomainForCast(DimensionIndex rank) {
  return StrCat("index domain with ",
                StaticCastTraits<DimensionIndex>::Describe(rank));
}

Result<IndexTransform<>> SliceByIndexDomain(IndexTransform<> transform,
                                            IndexDomainView<> domain) {
  using internal_index_space::TransformAccess;
  assert(transform.valid());
  assert(domain.valid());
  TransformRep::Ptr<> rep =
      MutableRep(TransformAccess::rep_ptr<container>(std::move(transform)));
  const DimensionIndex slice_rank = domain.rank();
  const DimensionIndex input_rank = rep->input_rank;
  const span<const std::string> domain_labels = domain.labels();
  const span<std::string> transform_labels =
      rep->input_labels().first(input_rank);
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims> transform_dims(
      slice_rank);
  const bool domain_unlabeled =
      internal_index_space::IsUnlabeled(domain_labels);
  if (domain_unlabeled || internal_index_space::IsUnlabeled(transform_labels)) {
    if (slice_rank != input_rank) {
      return Status(
          absl::StatusCode::kInvalidArgument,
          StrCat("Rank of index domain (", slice_rank,
                 ") must match rank of slice target (", input_rank,
                 ") when the index domain or slice target is unlabeled"));
    }
    std::iota(transform_dims.begin(), transform_dims.end(), DimensionIndex(0));
    if (!domain_unlabeled) {
      std::copy_n(domain_labels.begin(), slice_rank, transform_labels.begin());
    }
  } else {
    DimensionIndex next_potentially_unlabeled_dim = 0;
    for (DimensionIndex i = 0; i < slice_rank; ++i) {
      absl::string_view label = domain_labels[i];
      DimensionIndex j;
      if (!label.empty()) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            j, NormalizeDimensionLabel(label, transform_labels));
        // No need to check for duplicate labels, because the labels in `domain`
        // are already guaranteed to be unique.
      } else {
        while (true) {
          if (next_potentially_unlabeled_dim == input_rank) {
            return Status(
                absl::StatusCode::kInvalidArgument,
                "Number of unlabeled dimensions in index domain exceeds number "
                "of unlabeled dimensions in slice target");
          }
          if (transform_labels[next_potentially_unlabeled_dim].empty()) {
            j = next_potentially_unlabeled_dim++;
            break;
          }
          ++next_potentially_unlabeled_dim;
        }
      }
      transform_dims[i] = j;
    }
    if (next_potentially_unlabeled_dim != 0 && input_rank != slice_rank) {
      return Status(
          absl::StatusCode::kInvalidArgument,
          StrCat("Rank (", slice_rank,
                 ") of index domain containing unlabeled dimensions must "
                 "equal slice target rank (",
                 input_rank, ")"));
    }
  }
  for (DimensionIndex i = 0; i < slice_rank; ++i) {
    const DimensionIndex j = transform_dims[i];
    const internal_index_space::InputDimensionRef d = rep->input_dimension(j);
    const IndexInterval orig_domain =
        d.optionally_implicit_domain().effective_interval();
    const IndexInterval new_domain = domain[i];
    if (!Contains(orig_domain, new_domain)) {
      return Status(
          absl::StatusCode::kOutOfRange,
          StrCat("Cannot slice target dimension ", j, " {",
                 d.index_domain_dimension<view>(),
                 "} with index domain dimension ", i, " {", domain[i], "}"));
    }
    d.domain() = new_domain;
    d.implicit_lower_bound() = false;
    d.implicit_upper_bound() = false;
  }
  return TransformAccess::Make<IndexTransform<>>(std::move(rep));
}

}  // namespace internal_index_space

Result<bool> GetOutputRange(IndexTransformView<> transform,
                            MutableBoxView<> output_range) {
  ABSL_ASSERT(output_range.rank() == transform.output_rank());
  absl::FixedArray<bool, internal::kNumInlinedDims> input_dim_used(
      transform.input_rank(), false);
  bool exact = true;
  for (DimensionIndex output_dim = 0, output_rank = transform.output_rank();
       output_dim < output_rank; ++output_dim) {
    const auto output_index_map = transform.output_index_map(output_dim);
    const OutputIndexMethod method = output_index_map.stride() == 0
                                         ? OutputIndexMethod::constant
                                         : output_index_map.method();
    switch (method) {
      case OutputIndexMethod::constant: {
        TENSORSTORE_ASSIGN_OR_RETURN(
            output_range[output_dim],
            IndexInterval::Sized(output_index_map.offset(), 1));
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        const Index stride = output_index_map.stride();
        if (stride < -1 || stride > 1) exact = false;
        const DimensionIndex input_dim = output_index_map.input_dimension();
        // If more than one output dimension depends on a given input dimension
        // (i.e. the input dimension corresponds to the diagonal of two or more
        // output dimensions), then the output range is not exact.
        if (absl::exchange(input_dim_used[input_dim], true)) exact = false;
        TENSORSTORE_ASSIGN_OR_RETURN(
            output_range[output_dim],
            GetAffineTransformRange(transform.input_domain()[input_dim],
                                    output_index_map.offset(), stride));
        break;
      }
      case OutputIndexMethod::array: {
        // For an index array output index map, the output range is computed
        // based on the stored `index_range` rather than the actual indices in
        // the index array, and is always considered non-exact, even if the
        // index array happens to densely cover the `index_range`.
        exact = false;
        const auto index_array_ref = output_index_map.index_array();
        TENSORSTORE_ASSIGN_OR_RETURN(
            output_range[output_dim],
            GetAffineTransformRange(index_array_ref.index_range(),
                                    output_index_map.offset(),
                                    output_index_map.stride()));
        break;
      }
    }
  }
  return exact;
}

namespace internal_index_space {
Status ValidateInputDimensionResize(
    OptionallyImplicitIndexInterval input_domain, Index requested_inclusive_min,
    Index requested_exclusive_max) {
  if (requested_inclusive_min != kImplicit &&
      requested_inclusive_min != -kInfIndex &&
      !IsFiniteIndex(requested_inclusive_min)) {
    return absl::InvalidArgumentError(StrCat(
        "Invalid requested inclusive min value ", requested_inclusive_min));
  }
  if (requested_exclusive_max != kImplicit &&
      requested_exclusive_max != kInfIndex + 1 &&
      !IsFiniteIndex(requested_exclusive_max - 1)) {
    return absl::InvalidArgumentError(StrCat(
        "Invalid requested exclusive max value ", requested_exclusive_max));
  }
  if (requested_inclusive_min != kImplicit &&
      requested_exclusive_max != kImplicit &&
      requested_inclusive_min > requested_exclusive_max) {
    return absl::InvalidArgumentError(StrCat("Invalid requested bounds [",
                                             requested_inclusive_min, ", ",
                                             requested_exclusive_max, ")"));
  }
  if (!input_domain.implicit_lower() && requested_inclusive_min != kImplicit) {
    return absl::InvalidArgumentError("Cannot change explicit lower bound");
  }

  if (!input_domain.implicit_upper() && requested_exclusive_max != kImplicit) {
    return absl::InvalidArgumentError("Cannot change explicit upper bound");
  }
  return absl::OkStatus();
}
}  // namespace internal_index_space

Status PropagateInputDomainResizeToOutput(
    IndexTransformView<> transform,
    span<const Index> requested_input_inclusive_min,
    span<const Index> requested_input_exclusive_max,
    bool can_resize_tied_bounds, span<Index> output_inclusive_min_constraint,
    span<Index> output_exclusive_max_constraint,
    span<Index> new_output_inclusive_min, span<Index> new_output_exclusive_max,
    bool* is_noop) {
  ABSL_ASSERT(transform.valid());
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex output_rank = transform.output_rank();
  ABSL_ASSERT(requested_input_inclusive_min.size() == transform.input_rank());
  ABSL_ASSERT(requested_input_exclusive_max.size() == transform.input_rank());
  ABSL_ASSERT(output_inclusive_min_constraint.size() ==
              transform.output_rank());
  ABSL_ASSERT(output_exclusive_max_constraint.size() ==
              transform.output_rank());
  ABSL_ASSERT(new_output_inclusive_min.size() == transform.output_rank());
  ABSL_ASSERT(new_output_exclusive_max.size() == transform.output_rank());

  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    TENSORSTORE_RETURN_IF_ERROR(
        internal_index_space::ValidateInputDimensionResize(
            transform.input_domain()[input_dim],
            requested_input_inclusive_min[input_dim],
            requested_input_exclusive_max[input_dim]),
        MaybeAnnotateStatus(
            _,
            StrCat("Invalid resize request for input dimension ", input_dim)));
  }

  bool is_noop_value = true;

  // Handle `single_input_dimension` output index maps.  This is done first in
  // order to compute `is_noop_value`.  Validation of `constant` and `array`
  // output index maps is only done if `is_noop_value == false`.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    output_inclusive_min_constraint[output_dim] = kImplicit;
    output_exclusive_max_constraint[output_dim] = kImplicit;
    new_output_inclusive_min[output_dim] = kImplicit;
    new_output_exclusive_max[output_dim] = kImplicit;

    const auto map = transform.output_index_map(output_dim);
    if (map.method() != OutputIndexMethod::single_input_dimension) continue;
    const DimensionIndex input_dim = map.input_dimension();
    const Index requested_min = requested_input_inclusive_min[input_dim];
    const Index requested_max = requested_input_exclusive_max[input_dim];
    if (requested_min != kImplicit || requested_max != kImplicit) {
      is_noop_value = false;
      if (std::abs(map.stride()) != 1) {
        return absl::InvalidArgumentError(
            StrCat("Output dimension ", output_dim,
                   " depends on resized input dimension ", input_dim,
                   " with non-unit stride of ", map.stride()));
      }

      Result<OptionallyImplicitIndexInterval> output_bounds =
          GetAffineTransformRange(
              {IndexInterval::UncheckedHalfOpen(
                   requested_min == kImplicit ? -kInfIndex : requested_min,
                   requested_max == kImplicit ? kInfIndex + 1 : requested_max),
               requested_min == kImplicit, requested_max == kImplicit},
              map.offset(), map.stride());
      if (!output_bounds) {
        return MaybeAnnotateStatus(
            output_bounds.status(),
            StrCat("Error propagating bounds for output dimension ", output_dim,
                   " from requested bounds for input dimension ", input_dim));
      }
      if (!output_bounds->implicit_lower()) {
        new_output_inclusive_min[output_dim] = output_bounds->inclusive_min();
      }
      if (!output_bounds->implicit_upper()) {
        new_output_exclusive_max[output_dim] = output_bounds->exclusive_max();
      }
    }
  }

  *is_noop = is_noop_value;
  if (is_noop_value) return absl::OkStatus();

  // Number of output dimensions that depend on a given input dimension via a
  // `single_input_dimension` map.  Only used if
  // `can_resize_tied_bounds == false`.
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      num_input_dim_deps(input_rank, 0);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto map = transform.output_index_map(output_dim);
    switch (map.method()) {
      case OutputIndexMethod::constant:
        if (!IsFiniteIndex(map.offset())) {
          return Status(
              absl::StatusCode::kInvalidArgument,
              StrCat("Output dimension ", output_dim,
                     " has constant map with invalid offset ", map.offset()));
        }
        if (!can_resize_tied_bounds) {
          output_inclusive_min_constraint[output_dim] = map.offset();
          output_exclusive_max_constraint[output_dim] = map.offset() + 1;
        }
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        if (!can_resize_tied_bounds) {
          if (num_input_dim_deps[input_dim]++ != 0) {
            return absl::InvalidArgumentError(
                StrCat("Input dimension ", input_dim,
                       " corresponds to a diagonal but "
                       "`resize_tied_bounds` was not specified"));
          }
          if (std::abs(map.stride()) != 1) {
            return Status(
                absl::StatusCode::kInvalidArgument,
                StrCat("Output dimension ", output_dim,
                       " depends on input dimension ", input_dim,
                       " with non-unit stride of ", map.stride(),
                       " but `resize_tied_bounds` was not specified"));
          }

          Result<OptionallyImplicitIndexInterval> output_bounds =
              GetAffineTransformRange(transform.input_domain()[input_dim],
                                      map.offset(), map.stride());
          if (!output_bounds) {
            return MaybeAnnotateStatus(
                output_bounds.status(),
                StrCat("Error propagating bounds for output dimension ",
                       output_dim, " from existing bounds for input dimension ",
                       input_dim));
          }
          if (!output_bounds->implicit_lower()) {
            output_inclusive_min_constraint[output_dim] =
                output_bounds->inclusive_min();
          }
          if (!output_bounds->implicit_upper()) {
            output_exclusive_max_constraint[output_dim] =
                output_bounds->exclusive_max();
          }
        }
        break;
      }
      case OutputIndexMethod::array:
        // TODO(jbms): Consider treating rank-0 index array as constant map, and
        // maybe also handle other special cases (such as diagonal of size 1).
        if (!can_resize_tied_bounds) {
          return Status(
              absl::StatusCode::kInvalidArgument,
              StrCat("Output dimension ", output_dim,
                     " has index array map but `resize_tied_bounds` was "
                     "not specified"));
        }
        break;
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorstore
