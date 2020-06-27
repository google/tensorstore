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

#include "python/tensorstore/indexing_spec.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorstore/array.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/container_to_shared.h"
#include "tensorstore/rank.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

using pybind11::detail::npy_api;

/// Increments `indices` in C order relative to `shape`.
///
/// \dchecks `shape.size() == indices.size()`
void IncrementIndices(span<const Index> shape, span<Index> indices) {
  assert(shape.size() == indices.size());
  for (DimensionIndex i = indices.size() - 1; i >= 0; --i) {
    if (indices[i] + 1 < shape[i]) {
      ++indices[i];
      break;
    }
    indices[i] = 0;
  }
}

/// Returns an array of shape `{mask.rank(), N}` specifying the indices of the
/// true values in `mask`.
///
/// This is used to convert bool arrays in a NumPy-style indexing spec to the
/// index array representation used by `IndexTransform`.
SharedArray<Index> GetBoolTrueIndices(ArrayView<const bool> mask) {
  // TODO(jbms): Make this more efficient, possibly using some of the same
  // tricks as in NumPy (see array_boolean_subscript in
  // numpy/core/src/multiarray/mapping.c).
  std::vector<Index> indices;
  absl::FixedArray<Index, internal::kNumInlinedDims> cur_indices(mask.rank(),
                                                                 0);
  IterateOverArrays(
      [&](const bool* x) {
        if (*x) {
          indices.insert(indices.end(), cur_indices.begin(), cur_indices.end());
        }
        IncrementIndices(mask.shape(), cur_indices);
      },
      c_order, mask);
  const Index num_elements = indices.size() / mask.rank();
  return {internal::ContainerToSharedDataPointerWithOffset(std::move(indices)),
          {mask.rank(), num_elements},
          fortran_order};
}

/// Returns the number of dimensions to which an `Ellipsis` term would expand.
///
/// \param spec The indexing spec.
/// \param selection_rank The number of dimensions to which `spec` will be
///     applied.
/// \throws `py::index_error` if the result would be negative.
DimensionIndex GetNumEllipsisDims(const IndexingSpec& spec,
                                  DimensionIndex selection_rank) {
  const DimensionIndex num_ellipsis_dims =
      selection_rank - spec.num_output_dims - spec.num_new_dims;
  if (num_ellipsis_dims < 0 || (!spec.has_ellipsis && num_ellipsis_dims != 0)) {
    throw py::index_error(StrCat("Indexing expression requires ",
                                 spec.num_output_dims + spec.num_new_dims,
                                 " dimensions but selection has ",
                                 selection_rank, " dimensions"));
  }
  return num_ellipsis_dims;
}

/// Computes the mapping between the "intermediate" domain and the new "input"
/// domain that results from applying an `IndexingSpec` to an existing "output"
/// domain.
///
/// This is used by the overloads of `ToIndexTransform` for the cases where a
/// dimension selection is used, i.e. `usage != IndexingSpec::Usage::kDirect`.
///
/// When we apply an `IndexingSpec` which may contain `NewAxis` terms to an
/// existing "output" domain, there is an implicit "intermediate" domain that is
/// obtained from the "output" domain by inserting any new singleton dimensions
/// due to `NewAxis` terms but leaving all other dimensions as is.  (We refer to
/// the existing domain as the "output" domain because we will compute an
/// `IndexTransform` that maps from a new "input" domain to this existing
/// "output" domain.)
///
/// If `IndexingSpec` does not contain `NewAxis` terms, the "intermediate"
/// domain is equal to the "output" domain.
///
/// Any dimension indices specified in the input dimension selection are
/// evaluated with respect to this "intermediate" domain, rather than the
/// "output" domain, as otherwise it would be impossible to specify the position
/// of the new singleton dimensions.
///
/// \param spec The indexing spec, must have
///     `spec.usage != IndexingSpec::Usage::kDirect`.
/// \param intermediate_rank The rank of the "intermediate" domain.
/// \param selected_dims The resolved dimension selection to which `spec`
///     applies.  Each element must be in `[0, intermediate_rank)`.
/// \param indexed_input_dims[out] Array of length
///     `spec.num_input_dims + GetNumEllipsisDims(spec, selected_dims.size())`
///     specifying the sequence of dimensions of the new "input" domain
///     generated by the terms of `spec`, ordered by the order of the terms in
///     `spec`, not the numerical order in the "input" domain.  We call these
///     "indexed" input dimensions because they correspond to terms in the
///     `IndexingSpec`.
/// \param unindexed_input_dims[out] Array of length `input_rank` that maps each
///     dimension `input_dim` of the new "input" domain that is not in
///     `indexed_input_dims` to the corresponding "intermediate" dimension index
///     (these dimensions simply "pass through" unmodified).  We call these
///     "unindexed" input dimensions because they do not correspond to terms in
///     the `IndexingSpec`.  If `input_dim` is in `indexed_input_dims`, it maps
///     to `-1` instead.
void GetIndexedInputDims(const IndexingSpec& spec,
                         DimensionIndex intermediate_rank,
                         span<const DimensionIndex> selected_dims,
                         span<DimensionIndex> indexed_input_dims,
                         span<DimensionIndex> unindexed_input_dims) {
  const DimensionIndex num_ellipsis_dims =
      selected_dims.size() - (spec.num_output_dims + spec.num_new_dims);
  assert(num_ellipsis_dims >= 0);
  assert(indexed_input_dims.size() == spec.num_input_dims + num_ellipsis_dims);
  const DimensionIndex input_rank = unindexed_input_dims.size();
  std::fill_n(unindexed_input_dims.begin(), input_rank, DimensionIndex(-1));
  assert(input_rank == intermediate_rank + spec.num_input_dims -
                           spec.num_output_dims - spec.num_new_dims);

  // Number of dimensions in the new "input" domain corresponding to each
  // dimension of the "intermediate" domain, or `-1` for intermediate dimensions
  // not associated with any term in `spec`.  We compute this temporary array in
  // the loop below before computing the actual `indexed_input_dims` and
  // `unindexed_input_dims` maps because the order of terms in `spec` does not
  // necessarily match the order of dimensions in the "input" domain (due to the
  // reordering implied by the dimension selection in `selected_dims`).  This
  // array has size `intermediate_rank + 1`, in order to have sufficient space
  // for the final sum when we convert it to a cumulative sum array below.
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      input_dims_per_intermediate_dim(intermediate_rank + 1, -1);

  // Index into `selected_dims` of the next intermediate dimension not yet
  // consumed by prior terms of `spec`.
  DimensionIndex selected_dim_i = 0;
  bool joint_index_array_dims_remaining = spec.joint_index_arrays_consecutive;
  for (const auto& term : spec.terms) {
    if (std::holds_alternative<Index>(term)) {
      // Index terms consume an intermediate dimension and generate no new
      // dimension.
      input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] = 0;
      continue;
    }
    if (std::holds_alternative<IndexingSpec::NewAxis>(term)) {
      // NewAxis terms leave intermediate dimensions alone.
      input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] = 1;
      continue;
    }
    if (std::holds_alternative<IndexingSpec::Slice>(term)) {
      // Slice terms adjust but do not consume intermediate dimensions.
      input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] = 1;
      continue;
    }
    if (std::holds_alternative<IndexingSpec::Ellipsis>(term)) {
      // The Ellipsis term is equivalent to `num_ellipsis_dims` slice terms.
      for (DimensionIndex i = 0; i < num_ellipsis_dims; ++i) {
        input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] = 1;
      }
      continue;
    }
    if (auto* index_array = std::get_if<IndexingSpec::IndexArray>(&term)) {
      if (index_array->outer) {
        // Each array outer-indexed intermediate dimension correspond to
        // `index_array.rank()` input dimensions.
        input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] =
            index_array->index_array.rank();
      } else {
        // In `IndexingSpec::Mode::kDefault` mode, if
        // `spec.joint_index_arrays_consecutive == true`, the intermediate
        // dimension corresponding to the first index array term corresponds to
        // the `joint_index_array_shape` dimensions in the input domain.  Note
        // that the intermediate dimension corresponding to the first index
        // array term may not have the lowest dimension index within the
        // "intermediate" domain, since the dimension selection may change the
        // relative dimension order.
        input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] =
            joint_index_array_dims_remaining
                ? spec.joint_index_array_shape.size()
                : 0;
        joint_index_array_dims_remaining = false;
      }
      continue;
    }
    if (auto* bool_array = std::get_if<IndexingSpec::BoolArray>(&term)) {
      const DimensionIndex rank = bool_array->index_arrays.shape()[0];
      // This function is only used when
      // `usage != IndexingSpec::Usage::kDirect`, and in that case, zero-rank
      // boolean arrays are not supported in outer indexing mode and the
      // presence of a zero-rank boolean array disables the
      // `joint_index_arrays_consecutive` behavior.
      assert(rank != 0 ||
             (!bool_array->outer && !spec.joint_index_arrays_consecutive));
      // The boolean array applies to `rank` intermediate dimensions.  We
      // consider the first intermediate dimension (relative to the ordering
      // specified by the dimension selection, not the intermediate dimension
      // with the lowest dimension index) to correspond to the index array
      // dimension or dimensions.
      if (rank == 0) continue;
      input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] =
          // In outer indexing mode, the first intermediate dimension
          // corresponds to the single index array dimension corresponding to
          // the boolean array of the new "input" domain.
          bool_array->outer ? 1
                            // In vectorized indexing mode, the same behavior as
                            // for integer index arrays applies.
                            : joint_index_array_dims_remaining
                                  ? spec.joint_index_array_shape.size()
                                  : 0;
      if (!bool_array->outer) joint_index_array_dims_remaining = false;
      // Subsequent intermediate dimensions don't correspond to any dimension of
      // the new "input" domain.
      for (DimensionIndex i = 1; i < rank; ++i) {
        input_dims_per_intermediate_dim[selected_dims[selected_dim_i++]] = 0;
      }
    }
  }

  // Next dimension of the new "input" domain that has not yet been assigned.
  DimensionIndex input_dim = 0;

  // Next index into `indexed_input_dims` that has not yet been assigned.
  DimensionIndex indexed_input_dim_i = 0;

  if (!spec.joint_index_arrays_consecutive) {
    // The first `spec.joint_index_array_shape.size()` dimensions of the "input"
    // domain correspond to the joint index array shape.  By convention, we also
    // consider these the first "indexed" dimensions.
    for (DimensionIndex i = 0;
         i < static_cast<DimensionIndex>(spec.joint_index_array_shape.size());
         ++i) {
      indexed_input_dims[indexed_input_dim_i++] = input_dim++;
    }
  }

  // Convert `input_dims_per_intermediate_dim` in place to specify the first
  // "input" dimension corresponding to each "intermediate" dimension, i.e. the
  // cumulative sum starting at the current value of `input_dim`.  At the same
  // time, compute `unindexed_input_dims`.
  for (DimensionIndex intermediate_dim = 0;
       intermediate_dim < intermediate_rank; ++intermediate_dim) {
    DimensionIndex num_input_dims = std::exchange(
        input_dims_per_intermediate_dim[intermediate_dim], input_dim);
    if (num_input_dims == -1) {
      unindexed_input_dims[input_dim++] = intermediate_dim;
    } else {
      input_dim += num_input_dims;
    }
  }
  input_dims_per_intermediate_dim[intermediate_rank] = input_dim;

  // Compute `indexed_input_dims` by reodering `input_dims_per_intermediate_dim`
  // by `selected_dims`.
  for (const DimensionIndex intermediate_dim : selected_dims) {
    for (DimensionIndex
             input_dim = input_dims_per_intermediate_dim[intermediate_dim],
             end_input_dim =
                 input_dims_per_intermediate_dim[intermediate_dim + 1];
         input_dim < end_input_dim; ++input_dim) {
      indexed_input_dims[indexed_input_dim_i++] = input_dim;
    }
  }
}

/// If `spec` is a "scalar" term, normalize it by duplicating the term
/// `selection_rank` times.  Otherwise, return `spec` unchanged.
///
/// This handles the case where a single "scalar" term is specified to apply to
/// all dimensions in the dimension selection.
IndexingSpec GetNormalizedSpec(IndexingSpec spec,
                               DimensionIndex selection_rank) {
  if (spec.scalar) {
    auto term = spec.terms.front();
    spec.terms.resize(selection_rank, term);
    spec.num_input_dims *= selection_rank;
    spec.num_output_dims *= selection_rank;
    spec.num_new_dims *= selection_rank;
  }
  return spec;
}

/// Resolve the dimension selection `dim_selection` for the case of an
/// `IndexingSpec` that may contain `NewAxis` terms (for use as the first
/// operation of a dimension expression).
///
/// Dimensions specified by index or by `DimRangeSpec` are fully normalized to
/// intermediate dimension indices in `[0, intermediate_rank)`.
///
/// Dimensions specified by label are not fully normalized to intermediate
/// dimension indices, because that mapping cannot be determined at this stage.
/// Instead, they are represented as negative numbers `~output_dim`, where
/// `output_dim` is the dimension index in the "output" domain.
///
/// \param dim_selection The dimension selection to resolve.
/// \param intermediate_rank The rank of the "intermediate" domain.
/// \param labels The dimension labels of the existing "output" domain.
/// \param dimensions[out] Non-null pointer to buffer that will be set to the
///     resolved sequence of dimension indices.
/// \error `absl::StatusCode::kInvalidArgument` if `dim_selection` cannot be
///     resolved.
absl::Status GetPartiallyNormalizedIntermediateDims(
    span<const DynamicDimSpec> dim_selection, DimensionIndex intermediate_rank,
    span<const std::string> labels, DimensionIndexBuffer* dimensions) {
  dimensions->clear();
  for (const auto& dim_spec : dim_selection) {
    if (auto* s = std::get_if<std::string>(&dim_spec)) {
      TENSORSTORE_ASSIGN_OR_RETURN(const DimensionIndex dim,
                                   NormalizeDimensionLabel(*s, labels));
      dimensions->push_back(~dim);
    } else if (auto* index = std::get_if<DimensionIndex>(&dim_spec)) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          const DimensionIndex dim,
          NormalizeDimensionIndex(*index, intermediate_rank));
      dimensions->push_back(dim);
    } else {
      TENSORSTORE_RETURN_IF_ERROR(NormalizeDimRangeSpec(
          std::get<DimRangeSpec>(dim_spec), intermediate_rank, dimensions));
    }
  }
  return absl::OkStatus();
}

/// Converts an `IndexingSpec` to an `IndexTransform`.
///
/// This is the common implementation used by the public overloads of
/// `ToIndexTransform`.
///
/// \param spec The indexing spec.
/// \param output_space The "output" domain to which `spec` is applied.
/// \param indexed_output_dims The sequence of indices of "output" dimensions
///     consumed by terms (including `Ellipsis`) in `spec` (the order is given
///     by the order of terms in `spec`).
/// \param indexed_input_dims The sequence of indices of "input" dimensions
///     generated by terms (including `Ellipsis`) in `spec` (the order is given
///     by the order of terms in `spec`).
/// \param unindexed_input_dims Array of length `input_rank` that maps each
///     dimension `input_dim` of the new "input" domain that is not in
///     `indexed_input_dims` to the corresponding "intermediate" dimension index
///     (these dimensions simply "pass through" unmodified).  We call these
///     "unindexed" input dimensions because they do not correspond to terms in
///     the `IndexingSpec`.  If `input_dim` is in `indexed_input_dims`, it maps
///     to `-1` instead.
IndexTransform<> ToIndexTransform(
    const IndexingSpec& spec, IndexDomainView<> output_space,
    span<const DimensionIndex> indexed_output_dims,
    span<const DimensionIndex> indexed_input_dims,
    span<const DimensionIndex> unindexed_input_dims) {
  const DimensionIndex num_ellipsis_dims =
      indexed_output_dims.size() - spec.num_output_dims;
  const DimensionIndex input_rank = unindexed_input_dims.size();
  IndexTransformBuilder<> builder(input_rank, output_space.rank());
  DimensionIndex selected_input_dim_i = 0;
  DimensionIndex index_array_input_start_dim_i = -1;
  DimensionIndex selected_output_dim_i = 0;
  auto input_origin = builder.input_origin();
  auto input_shape = builder.input_shape();
  auto implicit_lower_bounds = builder.implicit_lower_bounds();
  auto implicit_upper_bounds = builder.implicit_upper_bounds();
  auto input_labels = builder.input_labels();

  const auto initialize_index_array_input_dimensions = [&] {
    index_array_input_start_dim_i = selected_input_dim_i;
    for (DimensionIndex i = 0;
         i < static_cast<DimensionIndex>(spec.joint_index_array_shape.size());
         ++i) {
      const DimensionIndex input_dim =
          indexed_input_dims[selected_input_dim_i++];
      implicit_lower_bounds[input_dim] = false;
      implicit_upper_bounds[input_dim] = false;
      input_origin[input_dim] = 0;
      input_shape[input_dim] = spec.joint_index_array_shape[i];
    }
  };
  if (!spec.joint_index_arrays_consecutive) {
    initialize_index_array_input_dimensions();
  }

  // Identity maps `input_dim` -> `output_dim`.  This is used for the `Ellipsis`
  // term and unindexed dimensions.
  const auto add_identity_map = [&](DimensionIndex input_dim,
                                    DimensionIndex output_dim) {
    const auto d = output_space[output_dim];
    builder.output_single_input_dimension(output_dim, input_dim);
    implicit_lower_bounds[input_dim] = d.implicit_lower();
    implicit_upper_bounds[input_dim] = d.implicit_upper();
    input_origin[input_dim] = d.inclusive_min();
    input_shape[input_dim] = d.size();
    input_labels[input_dim] = std::string(d.label());
  };

  const auto add_remaining_identity_maps = [&] {
    for (DimensionIndex i = 0; i < num_ellipsis_dims; ++i) {
      add_identity_map(indexed_input_dims[selected_input_dim_i++],
                       indexed_output_dims[selected_output_dim_i++]);
    }
  };

  const auto add_index_array = [&](const SharedArray<const Index>& array,
                                   DimensionIndex cur_input_start_dim_i) {
    SharedArray<const Index> broadcast_array;
    broadcast_array.layout().set_rank(input_rank);
    std::fill_n(broadcast_array.byte_strides().begin(), input_rank, Index(0));
    std::fill_n(broadcast_array.shape().begin(), input_rank, Index(1));
    for (DimensionIndex i = 0; i < array.rank(); ++i) {
      const DimensionIndex input_dim =
          indexed_input_dims[cur_input_start_dim_i + i];
      broadcast_array.byte_strides()[input_dim] = array.byte_strides()[i];
      broadcast_array.shape()[input_dim] = array.shape()[i];
    }
    broadcast_array.element_pointer() = array.element_pointer();
    builder.output_index_array(indexed_output_dims[selected_output_dim_i], 0, 1,
                               std::move(broadcast_array));
    ++selected_output_dim_i;
  };

  const auto add_index_array_domain = [&](span<const Index> shape,
                                          bool outer) -> DimensionIndex {
    if (outer) {
      DimensionIndex cur_input_start_dim_i = selected_input_dim_i;
      for (DimensionIndex i = 0; i < shape.size(); ++i) {
        const DimensionIndex input_dim =
            indexed_input_dims[selected_input_dim_i++];
        input_origin[input_dim] = 0;
        input_shape[input_dim] = shape[i];
        implicit_lower_bounds[input_dim] = false;
        implicit_upper_bounds[input_dim] = false;
      }
      return cur_input_start_dim_i;
    }
    if (index_array_input_start_dim_i == -1) {
      initialize_index_array_input_dimensions();
    }
    return index_array_input_start_dim_i + spec.joint_index_array_shape.size() -
           shape.size();
  };

  for (DimensionIndex input_dim = 0; input_dim < unindexed_input_dims.size();
       ++input_dim) {
    const DimensionIndex output_dim = unindexed_input_dims[input_dim];
    if (output_dim != -1) {
      add_identity_map(input_dim, output_dim);
    }
  }

  for (const auto& term : spec.terms) {
    if (std::holds_alternative<IndexingSpec::Ellipsis>(term)) {
      add_remaining_identity_maps();
      continue;
    }
    if (std::holds_alternative<IndexingSpec::NewAxis>(term)) {
      const DimensionIndex input_dim =
          indexed_input_dims[selected_input_dim_i++];
      input_origin[input_dim] = 0;
      input_shape[input_dim] = 1;
      implicit_lower_bounds[input_dim] = true;
      implicit_upper_bounds[input_dim] = true;
      continue;
    }

    if (auto* s = std::get_if<IndexingSpec::Slice>(&term)) {
      const DimensionIndex input_dim =
          indexed_input_dims[selected_input_dim_i++];
      const DimensionIndex output_dim =
          indexed_output_dims[selected_output_dim_i++];
      const auto d = output_space[output_dim];
      OptionallyImplicitIndexInterval new_domain;
      Index offset;
      auto status = ComputeStridedSliceMap(
          d.optionally_implicit_interval(), IntervalForm::half_open,
          /*translate_origin_to=*/kImplicit, s->start, s->stop, s->step,
          &new_domain, &offset);
      if (!status.ok()) {
        throw py::index_error(StrCat("Computing interval slice for dimension ",
                                     output_dim, ": ", status.message()));
      }
      implicit_lower_bounds[input_dim] = new_domain.implicit_lower();
      implicit_upper_bounds[input_dim] = new_domain.implicit_upper();
      input_origin[input_dim] = new_domain.inclusive_min();
      input_shape[input_dim] = new_domain.size();
      input_labels[input_dim] = std::string(d.label());
      builder.output_single_input_dimension(output_dim, offset, s->step,
                                            input_dim);
      continue;
    }

    if (auto* index = std::get_if<Index>(&term)) {
      const DimensionIndex output_dim =
          indexed_output_dims[selected_output_dim_i++];
      builder.output_constant(output_dim, *index);
      continue;
    }

    if (auto* bool_array = std::get_if<IndexingSpec::BoolArray>(&term)) {
      const DimensionIndex rank = bool_array->index_arrays.shape()[0];
      const DimensionIndex cur_input_start_dim_i = add_index_array_domain(
          bool_array->index_arrays.shape().subspan(1), bool_array->outer);
      for (DimensionIndex i = 0; i < rank; ++i) {
        add_index_array(
            SharedSubArray<container>(bool_array->index_arrays, {i}),
            cur_input_start_dim_i);
      }
      continue;
    }

    // Remaining case is `IndexingSpec::IndexArray`.
    const auto& index_array = std::get<IndexingSpec::IndexArray>(term);
    const DimensionIndex cur_input_start_dim_i = add_index_array_domain(
        index_array.index_array.shape(), index_array.outer);
    add_index_array(index_array.index_array, cur_input_start_dim_i);
  }

  if (!spec.has_ellipsis) {
    add_remaining_identity_maps();
  }
  return ValueOrThrow(builder.Finalize(), StatusExceptionPolicy::kIndexError);
}

/// Returns `py::cast<T>(handle)`, but throws an exception that maps to a Python
/// `TypeError` exception with a message of `msg` (as is typical for Python
/// APIs), rather than the pybind11-specific `py::cast_error`.
template <typename T>
T CastOrTypeError(py::handle handle, const char* msg) {
  try {
    return py::cast<T>(handle);
  } catch (py::cast_error&) {
    throw py::type_error(msg);
  }
}

}  // namespace

absl::string_view GetIndexingModePrefix(IndexingSpec::Mode mode) {
  switch (mode) {
    case IndexingSpec::Mode::kDefault:
      return "";
    case IndexingSpec::Mode::kOindex:
      return ".oindex";
    case IndexingSpec::Mode::kVindex:
      return ".vindex";
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

IndexingSpec IndexingSpec::Parse(pybind11::handle obj, IndexingSpec::Mode mode,
                                 IndexingSpec::Usage usage) {
  IndexingSpec spec;
  spec.mode = mode;
  spec.usage = usage;
  spec.scalar = true;
  spec.has_ellipsis = false;
  spec.num_output_dims = 0;
  spec.num_input_dims = 0;
  spec.num_new_dims = 0;
  spec.joint_index_arrays_consecutive = mode == IndexingSpec::Mode::kDefault;
  auto& api = npy_api::get();

  bool has_index_array = false;
  bool has_index_array_break = false;

  const auto add_index_array_shape = [&](span<const Index> shape) {
    if (mode == IndexingSpec::Mode::kOindex) {
      spec.num_input_dims += shape.size();
      return;
    }
    if (static_cast<DimensionIndex>(spec.joint_index_array_shape.size()) <
        shape.size()) {
      spec.joint_index_array_shape.insert(
          spec.joint_index_array_shape.begin(),
          shape.size() - spec.joint_index_array_shape.size(), 1);
    }
    for (DimensionIndex i = 0; i < shape.size(); ++i) {
      const Index size = shape[i];
      Index& broadcast_size =
          spec.joint_index_array_shape[spec.joint_index_array_shape.size() -
                                       (shape.size() - i)];
      if (size != 1) {
        if (broadcast_size != 1 && broadcast_size != size) {
          throw py::index_error(
              StrCat("Incompatible index array shapes: ", shape, " vs ",
                     span(spec.joint_index_array_shape)));
        }
        broadcast_size = size;
      }
    }
    has_index_array = true;
    if (has_index_array_break) {
      spec.joint_index_arrays_consecutive = false;
    }
  };

  const auto add_index_array = [&](SharedArray<const Index> index_array) {
    add_index_array_shape(index_array.shape());
    ++spec.num_output_dims;
    spec.terms.emplace_back(IndexingSpec::IndexArray{
        std::move(index_array), mode == IndexingSpec::Mode::kOindex});
  };

  // Process a Python object representing a single indexing term.  This
  // conversion mostly follows the logic in numpy/core/src/multiarray/mapping.c
  // for compatibility with NumPy.
  const auto add_term = [&](py::handle term) {
    if (term.ptr() == Py_Ellipsis) {
      if (spec.has_ellipsis) {
        throw py::index_error(
            "An index can only have a single ellipsis (`...`)");
      }
      spec.scalar = false;
      spec.terms.emplace_back(IndexingSpec::Ellipsis{});
      spec.has_ellipsis = true;
      has_index_array_break = has_index_array;
      return;
    }
    if (term.ptr() == Py_None) {
      if (usage == IndexingSpec::Usage::kDimSelectionChained) {
        throw py::index_error(
            "tensorstore.newaxis (`None`) not valid in chained indexing "
            "operations");
      }
      ++spec.num_input_dims;
      ++spec.num_new_dims;
      spec.terms.emplace_back(IndexingSpec::NewAxis{});
      has_index_array_break = has_index_array;
      return;
    }
    if (PySlice_Check(term.ptr())) {
      auto* slice_obj = reinterpret_cast<PySliceObject*>(term.ptr());
      const auto get_slice_index = [](py::handle handle) {
        return ToIndexVectorOrScalarContainer(
            CastOrTypeError<OptionallyImplicitIndexVectorOrScalarContainer>(
                handle,
                "slice indices must be integers or None or have an __index__ "
                "method"));
      };
      auto start = get_slice_index(slice_obj->start);
      auto stop = get_slice_index(slice_obj->stop);
      auto step = get_slice_index(slice_obj->step);
      DimensionIndex rank = dynamic_rank;
      {
        const IndexVectorOrScalarContainer* existing_value = nullptr;
        const char* existing_field_name = nullptr;
        const auto check_rank = [&](const IndexVectorOrScalarContainer& x,
                                    const char* field_name) {
          if (auto* v = std::get_if<std::vector<Index>>(&x)) {
            if (rank != dynamic_rank &&
                rank != static_cast<DimensionIndex>(v->size())) {
              throw py::index_error(
                  StrCat(field_name, "=", IndexVectorRepr(x, /*implicit=*/true),
                         " (rank ", v->size(), ") is incompatible with ",
                         existing_field_name, "=",
                         IndexVectorRepr(*existing_value, /*implicit=*/true),
                         " (rank ", rank, ")"));
            }
            existing_field_name = field_name;
            rank = v->size();
            existing_value = &x;
          }
        };
        check_rank(start, "start");
        check_rank(stop, "stop");
        check_rank(step, "step");
      }
      if (rank != dynamic_rank) {
        spec.scalar = false;
      } else {
        rank = 1;
      }
      for (DimensionIndex i = 0; i < rank; ++i) {
        Index step_value = ToIndexVectorOrScalar(step)[i];
        if (step_value == kImplicit) step_value = 1;
        spec.terms.emplace_back(
            IndexingSpec::Slice{ToIndexVectorOrScalar(start)[i],
                                ToIndexVectorOrScalar(stop)[i], step_value});
      }
      spec.num_input_dims += rank;
      spec.num_output_dims += rank;
      has_index_array_break = has_index_array;
      return;
    }
    // Check for an integer index.  Bool scalars are not treated as integer
    // indices; instead, they are treated as rank-0 boolean arrays.
    if (PyLong_CheckExact(term.ptr()) ||
        (!PyBool_Check(term.ptr()) && !api.PyArray_Check_(term.ptr()))) {
      ssize_t x = PyNumber_AsSsize_t(term.ptr(), PyExc_IndexError);
      if (x != -1 || !PyErr_Occurred()) {
        spec.terms.emplace_back(static_cast<Index>(x));
        ++spec.num_output_dims;
        return;
      }
      PyErr_Clear();
    }

    py::array array_obj;

    // Only remaining cases are index arrays, bool arrays, or invalid values.
    spec.scalar = false;

    if (!api.PyArray_Check_(term.ptr())) {
      array_obj = py::reinterpret_steal<py::array>(api.PyArray_FromAny_(
          term.ptr(), nullptr, 0, 0, npy_api::NPY_ARRAY_ALIGNED_, nullptr));
      if (!array_obj) throw py::error_already_set();
      if (array_obj.size() == 0) {
        array_obj = py::reinterpret_steal<py::array>(api.PyArray_FromAny_(
            array_obj.ptr(), GetNumpyDtype<Index>().release().ptr(), 0, 0,
            npy_api::NPY_ARRAY_FORCECAST_ | npy_api::NPY_ARRAY_ALIGNED_,
            nullptr));
        if (!array_obj) throw py::error_already_set();
      }
    } else {
      array_obj = py::reinterpret_borrow<py::array>(term.ptr());
    }

    auto* array_proxy = py::detail::array_proxy(array_obj.ptr());
    const int type_num =
        py::detail::array_descriptor_proxy(array_proxy->descr)->type_num;
    if (type_num == npy_api::NPY_BOOL_) {
      // Bool array.
      auto array = UncheckedArrayFromNumpy<bool>(std::move(array_obj));
      SharedArray<const Index> index_arrays;
      if (array.rank() == 0) {
        if (usage != IndexingSpec::Usage::kDirect) {
          if (mode == IndexingSpec::Mode::kOindex) {
            throw py::index_error(
                "Zero-rank bool array incompatible with outer indexing of a "
                "dimension selection");
          } else {
            spec.joint_index_arrays_consecutive = false;
          }
        }
        // Rank 0: corresponds to a dummy dimension of length 0 or 1
        index_arrays.layout() = StridedLayout<2>({0, array() ? 1 : 0}, {0, 0});
      } else {
        index_arrays = GetBoolTrueIndices(array);
      }
      spec.num_output_dims += array.rank();
      add_index_array_shape(index_arrays.shape().subspan(1));
      spec.terms.emplace_back(IndexingSpec::BoolArray{
          std::move(index_arrays), mode == IndexingSpec::Mode::kOindex});
      return;
    }
    if (type_num >= npy_api::NPY_BYTE_ && type_num <= npy_api::NPY_ULONGLONG_) {
      // Integer array.
      array_obj = py::reinterpret_steal<py::array>(api.PyArray_FromAny_(
          array_obj.ptr(), GetNumpyDtype<Index>().release().ptr(), 0, 0,
          npy_api::NPY_ARRAY_ALIGNED_, nullptr));
      if (!array_obj) {
        throw py::error_already_set();
      }
      // TODO(jbms): Add mechanism for users to explicitly indicate that an
      // index array can safely be stored by reference rather than copied.  User
      // must ensure that array is not modified.
      add_index_array(MakeCopy(UncheckedArrayFromNumpy<Index>(array_obj),
                               skip_repeated_elements));
      return;
    }
    // Invalid array data type.
    if (array_obj.ptr() == term.ptr()) {
      // The input was already an array.
      throw py::index_error(
          "Arrays used as indices must be of integer (or boolean) type");
    }
    throw py::index_error(
        "Only integers, slices (`:`), ellipsis (`...`), tensorstore.newaxis "
        "(`None`) and integer or boolean arrays are valid indices");
  };

  if (!PyTuple_Check(obj.ptr())) {
    add_term(obj);
  } else {
    spec.scalar = false;
    py::tuple t = py::reinterpret_borrow<py::tuple>(obj);
    for (size_t i = 0, size = t.size(); i < size; ++i) {
      add_term(t[i]);
    }
  }
  spec.num_input_dims += spec.joint_index_array_shape.size();
  return spec;
}

SharedArray<bool> GetBoolArrayFromIndices(
    ArrayView<const Index, 2> index_arrays) {
  const DimensionIndex rank = index_arrays.shape()[0];
  absl::FixedArray<Index, internal::kNumInlinedDims> shape(rank);
  const Index num_indices = index_arrays.shape()[1];
  for (DimensionIndex j = 0; j < rank; ++j) {
    Index x = 0;
    for (Index i = 0; i < num_indices; ++i) {
      x = std::max(x, index_arrays(j, i));
    }
    shape[j] = x + 1;
  }
  auto bool_array = AllocateArray<bool>(shape, c_order, value_init);
  for (Index i = 0; i < num_indices; ++i) {
    Index offset = 0;
    for (DimensionIndex j = 0; j < rank; ++j) {
      offset += bool_array.byte_strides()[j] * index_arrays(j, i);
    }
    bool_array.byte_strided_pointer()[offset] = true;
  }
  return bool_array;
}

std::string IndexingSpec::repr() const {
  std::string r;
  for (size_t i = 0; i < terms.size(); ++i) {
    if (i != 0) r += ",";
    const auto& term = terms[i];
    if (auto* index = std::get_if<Index>(&term)) {
      StrAppend(&r, *index);
      continue;
    }
    if (auto* s = std::get_if<IndexingSpec::Slice>(&term)) {
      if (s->start != kImplicit) StrAppend(&r, s->start);
      r += ':';
      if (s->stop != kImplicit) StrAppend(&r, s->stop);
      if (s->step != 1) StrAppend(&r, ":", s->step);
      continue;
    }
    if (std::holds_alternative<IndexingSpec::NewAxis>(term)) {
      r += "None";
      continue;
    }
    if (std::holds_alternative<IndexingSpec::Ellipsis>(term)) {
      r += "...";
      continue;
    }
    if (auto* index_array = std::get_if<IndexingSpec::IndexArray>(&term)) {
      r += py::repr(py::cast(index_array->index_array));
      continue;
    }
    if (auto* bool_array = std::get_if<IndexingSpec::BoolArray>(&term)) {
      r += py::repr(py::cast(GetBoolArrayFromIndices(
          StaticRankCast<2, unchecked>(bool_array->index_arrays))));
    }
  }
  if (!scalar && terms.size() == 1) {
    r += ',';
  }
  return r;
}

IndexTransform<> ToIndexTransform(const IndexingSpec& spec,
                                  IndexDomainView<> output_space) {
  const DimensionIndex output_rank = output_space.rank();
  assert(spec.usage == IndexingSpec::Usage::kDirect);
  if (spec.num_output_dims > output_rank) {
    throw py::index_error(
        StrCat("Indexing expression requires ", spec.num_output_dims,
               " dimensions, and cannot be applied to a domain of rank ",
               output_rank));
  }
  const DimensionIndex num_ellipsis_dims = output_rank - spec.num_output_dims;
  const DimensionIndex input_rank = spec.num_input_dims + num_ellipsis_dims;
  DimensionIndexBuffer indexed_input_dims, indexed_output_dims;
  indexed_input_dims.resize(input_rank);
  std::iota(indexed_input_dims.begin(), indexed_input_dims.end(),
            DimensionIndex(0));
  indexed_output_dims.resize(output_rank);
  std::iota(indexed_output_dims.begin(), indexed_output_dims.end(),
            DimensionIndex(0));
  return ToIndexTransform(spec, output_space, indexed_output_dims,
                          indexed_input_dims,
                          GetConstantVector<DimensionIndex, -1>(input_rank));
}

IndexTransform<> ToIndexTransform(IndexingSpec spec,
                                  IndexDomainView<> output_space,
                                  DimensionIndexBuffer* dimensions) {
  assert(spec.num_new_dims == 0);
  assert(spec.usage == IndexingSpec::Usage::kDimSelectionChained);
  spec = GetNormalizedSpec(std::move(spec), dimensions->size());
  const DimensionIndex num_ellipsis_dims =
      GetNumEllipsisDims(spec, dimensions->size());
  DimensionIndexBuffer indexed_input_dims(spec.num_input_dims +
                                          num_ellipsis_dims);
  const DimensionIndex output_rank = output_space.rank();
  const DimensionIndex input_rank =
      spec.num_input_dims + output_rank - dimensions->size();
  DimensionIndexBuffer unindexed_input_dims(input_rank);
  GetIndexedInputDims(spec, output_rank, *dimensions, indexed_input_dims,
                      unindexed_input_dims);
  auto transform = ToIndexTransform(spec, output_space, *dimensions,
                                    indexed_input_dims, unindexed_input_dims);
  *dimensions = std::move(indexed_input_dims);
  return transform;
}

IndexTransform<> ToIndexTransform(IndexingSpec spec,
                                  IndexDomainView<> output_space,
                                  span<const DynamicDimSpec> dim_selection,
                                  DimensionIndexBuffer* dimensions) {
  assert(spec.usage == IndexingSpec::Usage::kDimSelectionInitial);
  DimensionIndex intermediate_rank;
  if (spec.scalar && spec.num_new_dims == 1) {
    ThrowStatusException(internal_index_space::GetNewDimensions(
                             output_space.rank(), dim_selection, dimensions),
                         StatusExceptionPolicy::kIndexError);
    intermediate_rank = output_space.rank() + dimensions->size();
  } else {
    intermediate_rank = output_space.rank() + spec.num_new_dims;
    ThrowStatusException(GetPartiallyNormalizedIntermediateDims(
                             dim_selection, intermediate_rank,
                             output_space.labels(), dimensions),
                         StatusExceptionPolicy::kIndexError);
  }

  absl::FixedArray<bool, internal::kNumInlinedDims>
      selected_intermediate_dim_mask(intermediate_rank, false);

  const auto check_for_duplicate_intermediate_dim = [&](DimensionIndex x) {
    auto& m = selected_intermediate_dim_mask[x];
    if (m == true) {
      throw py::index_error(
          StrCat("Dimension ", x, " specified more than once"));
    }
    m = true;
  };

  for (auto x : *dimensions) {
    if (x < 0) continue;
    check_for_duplicate_intermediate_dim(x);
  }

  spec = GetNormalizedSpec(std::move(spec), dimensions->size());
  const DimensionIndex num_ellipsis_dims =
      GetNumEllipsisDims(spec, dimensions->size());
  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      intermediate_to_output(intermediate_rank, 0);
  {
    DimensionIndex selected_dim_i = 0;
    for (const auto& term : spec.terms) {
      if (std::holds_alternative<Index>(term)) {
        ++selected_dim_i;
        continue;
      }
      if (std::holds_alternative<IndexingSpec::Slice>(term)) {
        ++selected_dim_i;
        continue;
      }
      if (std::holds_alternative<IndexingSpec::Ellipsis>(term)) {
        selected_dim_i += num_ellipsis_dims;
        continue;
      }
      if (std::holds_alternative<IndexingSpec::NewAxis>(term)) {
        const DimensionIndex intermediate_dim = (*dimensions)[selected_dim_i];
        if (intermediate_dim < 0) {
          throw py::index_error(
              "Dimensions specified by label cannot be used with newaxis");
        }
        intermediate_to_output[intermediate_dim] = -1;
        ++selected_dim_i;
        continue;
      }
      if (std::holds_alternative<IndexingSpec::IndexArray>(term)) {
        ++selected_dim_i;
        continue;
      }
      if (auto* bool_array = std::get_if<IndexingSpec::BoolArray>(&term)) {
        selected_dim_i += bool_array->index_arrays.shape()[0];
        continue;
      }
    }
    assert(selected_dim_i == static_cast<DimensionIndex>(dimensions->size()));
  }

  absl::FixedArray<DimensionIndex, internal::kNumInlinedDims>
      output_to_intermediate(output_space.rank());
  {
    DimensionIndex output_dim = 0;
    for (DimensionIndex intermediate_dim = 0;
         intermediate_dim < intermediate_rank; ++intermediate_dim) {
      auto& dim = intermediate_to_output[intermediate_dim];
      if (dim == -1) continue;
      output_to_intermediate[output_dim] = intermediate_dim;
      dim = output_dim++;
    }
    assert(output_dim == output_space.rank());
  }

  for (auto& x : *dimensions) {
    if (x >= 0) continue;
    x = output_to_intermediate[-(x + 1)];
    check_for_duplicate_intermediate_dim(x);
  }

  DimensionIndexBuffer indexed_input_dims(spec.num_input_dims +
                                          num_ellipsis_dims);
  DimensionIndexBuffer unindexed_input_dims(
      output_space.rank() + spec.num_input_dims - spec.num_output_dims);
  GetIndexedInputDims(spec, intermediate_rank, *dimensions, indexed_input_dims,
                      unindexed_input_dims);
  for (auto& x : unindexed_input_dims) {
    if (x == -1) continue;
    assert(x >= 0 && x < intermediate_rank);
    x = intermediate_to_output[x];
    assert(x >= 0);
  }
  for (auto& x : *dimensions) {
    assert(x >= 0 && x < intermediate_rank);
    x = intermediate_to_output[x];
  }
  dimensions->erase(
      std::remove(dimensions->begin(), dimensions->end(), DimensionIndex(-1)),
      dimensions->end());
  auto transform = ToIndexTransform(spec, output_space, *dimensions,
                                    indexed_input_dims, unindexed_input_dims);
  *dimensions = std::move(indexed_input_dims);
  return transform;
}

}  // namespace internal_python
}  // namespace tensorstore
