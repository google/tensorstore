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

#include "tensorstore/index_space/internal/propagate_bounds.h"

#include <sstream>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "tensorstore/index_space/internal/identity_transform.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {
namespace {

absl::Status PropagateBoundsImpl(BoxView<> b,
                                 DimensionSet b_implicit_lower_bounds,
                                 DimensionSet b_implicit_upper_bounds,
                                 TransformRep* a_to_b, MutableBoxView<> a) {
  if (!a_to_b) {
    // A null `a_to_b` pointer indicates an identity transform, which we must
    // handle specially: the propagated bounds for `a` are simply equal to the
    // specified bounds for `b`.
    assert(a.rank() == b.rank());
    a.DeepAssign(b);
    return absl::OkStatus();
  }

  assert(a_to_b->input_rank == a.rank());
  assert(a_to_b->output_rank == b.rank());

  // Initialize the propagated bounds for `a` to the unbounded interval.  We
  // will iteratively reduce the bounds for input dimensions by intersecting
  // them with the inferred bounds from corresponding dimensions of `b`.
  a.Fill();

  span<const OutputIndexMap> maps = a_to_b->output_index_maps().first(b.rank());

  // Specifies whether bounds from `b` were propagated for a given dimension of
  // `a`.  If not, the original bounds are retained.
  absl::FixedArray<bool, internal::kNumInlinedDims> propagated_to_a(a.rank(),
                                                                    false);

  DimensionSet inferred_implicit_lower_bounds(true);
  DimensionSet inferred_implicit_upper_bounds(true);

  auto& implicit_lower_bounds = a_to_b->implicit_lower_bounds;
  auto& implicit_upper_bounds = a_to_b->implicit_upper_bounds;
  const auto existing_input_domain = a_to_b->input_domain(a.rank());

  // Determine if the domain is empty.  If the domain is empty, bounds checking
  // is skipped.
  bool is_domain_empty = false;
  for (DimensionIndex a_dim = 0; a_dim < a.rank(); ++a_dim) {
    if (!implicit_lower_bounds[a_dim] && !implicit_upper_bounds[a_dim] &&
        existing_input_domain[a_dim].empty()) {
      is_domain_empty = true;
      break;
    }
  }

  // Validate the output index map for each `b` dimension, and for
  // single_input_dimension maps also updates the propagated bounds for the
  // corresponding `a` dimension.
  for (DimensionIndex b_dim = 0; b_dim < b.rank(); ++b_dim) {
    auto& map = maps[b_dim];
    const Index output_stride = map.stride();
    // We dont't validate or propagate bounds for array-based output index maps.
    if (map.method() == OutputIndexMethod::array) continue;
    OptionallyImplicitIndexInterval b_bounds_oi{b[b_dim],
                                                b_implicit_lower_bounds[b_dim],
                                                b_implicit_upper_bounds[b_dim]};
    if (output_stride == 0 || map.method() == OutputIndexMethod::constant) {
      if (!is_domain_empty) {
        // Validate the constant output index map.  We can't propagate bounds to
        // `a` in this case.
        TENSORSTORE_RETURN_IF_ERROR(
            CheckContains(b_bounds_oi.effective_interval(), map.offset()),
            MaybeAnnotateStatus(
                _, tensorstore::StrCat("Checking bounds of constant output "
                                       "index map for dimension ",
                                       b_dim)));
      }
      continue;
    }
    const DimensionIndex a_dim = map.input_dimension();
    assert(a_dim >= 0 && a_dim < a.rank());
    // TODO(jbms): In some cases, this will result in a result_out_of_range
    // error even though we could have obtained a non-overflowed bound from a
    // different dimension, or from the input domain.
    TENSORSTORE_ASSIGN_OR_RETURN(
        OptionallyImplicitIndexInterval propagated_a_bounds,
        GetAffineTransformDomain(b_bounds_oi, map.offset(), map.stride()),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Propagating bounds from dimension ", b_dim,
                                   " to input dimension ", a_dim)));
    propagated_a_bounds = IntersectPreferringExplicit(
        propagated_a_bounds,
        OptionallyImplicitIndexInterval{a[a_dim],
                                        inferred_implicit_lower_bounds[a_dim],
                                        inferred_implicit_upper_bounds[a_dim]});
    a[a_dim] = propagated_a_bounds.interval();
    inferred_implicit_lower_bounds[a_dim] =
        propagated_a_bounds.implicit_lower();
    inferred_implicit_upper_bounds[a_dim] =
        propagated_a_bounds.implicit_upper();
    propagated_to_a[a_dim] = true;
  }

  // Validate that the inferred bounds are compatible with the bounds specified
  // in the `a_to_b` transform.
  for (DimensionIndex a_dim = 0; a_dim < a.rank(); ++a_dim) {
    IndexInterval existing = existing_input_domain[a_dim];
    IndexIntervalRef inferred = a[a_dim];
    if (!propagated_to_a[a_dim]) {
      inferred = existing;
      continue;
    }
    const Index inclusive_min = implicit_lower_bounds[a_dim]
                                    ? inferred.inclusive_min()
                                    : existing.inclusive_min();
    const Index inclusive_max =
        std::max(inclusive_min - 1, implicit_upper_bounds[a_dim]
                                        ? inferred.inclusive_max()
                                        : existing.inclusive_max());
    const IndexInterval combined =
        IndexInterval::UncheckedClosed(inclusive_min, inclusive_max);
    const OptionallyImplicitIndexInterval inferred_oi{
        inferred, inferred_implicit_lower_bounds[a_dim],
        inferred_implicit_upper_bounds[a_dim]};
    if (!is_domain_empty &&
        !Contains(inferred_oi.effective_interval(), combined)) {
      std::ostringstream os;
      os << "Propagated bounds " << inferred_oi;
      if (inferred_oi.size() != kInfSize) {
        os << ", with size=" << inferred_oi.size() << ", ";
      }
      os << "for dimension " << a_dim
         << " are incompatible with existing bounds " << combined;
      if (combined.size() != kInfSize) {
        os << ", with size=" << combined.size();
      }
      os << ".";
      return absl::OutOfRangeError(os.str());
    }
    inferred = combined;
  }
  return absl::OkStatus();
}

/// Propagates the implicit lower/upper bound bits from "b" to "a".
///
/// Each propagated lower/upper bound for "a" is implicit iff:
///
/// 1. The original bound specified in `a_to_b` is implicit; and
///
/// 2. All contributing (by way of a `single_input_dimension` map) bounds of "b"
///    are implicit.
///
/// \param b_rank The rank of the output index space "b".
/// \param b_implicit_lower_bounds Implicit indicator for each lower bound of
///     "b".
/// \param b_implicit_upper_bounds Implicit indicator for each upper bound of
///     "b".
/// \param a_to_b[in] The transform, or `nullptr` to indicate an identity
///     transform.
/// \param a_rank The rank of the input index space "a".
/// \param a_implicit_lower_bounds[out] Propagated implicit indicators for each
///     lower bound of "a".  May alias `a_to_b->implicit_lower_bounds`.
/// \param a_implicit_upper_bounds[out] Propagated implicit indicators for each
///     upper bound of "a".  May alias `a_to_b->implicit_upper_bounds`.
void PropagateImplicitBoundState(DimensionIndex b_rank,
                                 DimensionSet b_implicit_lower_bounds,
                                 DimensionSet b_implicit_upper_bounds,
                                 TransformRep* a_to_b, DimensionIndex a_rank,
                                 DimensionSet& a_implicit_lower_bounds,
                                 DimensionSet& a_implicit_upper_bounds) {
  if (!a_to_b) {
    a_implicit_lower_bounds = b_implicit_lower_bounds;
    a_implicit_upper_bounds = b_implicit_upper_bounds;
    return;
  }
  a_implicit_lower_bounds = a_to_b->implicit_lower_bounds;
  a_implicit_upper_bounds = a_to_b->implicit_upper_bounds;
  span<const OutputIndexMap> maps = a_to_b->output_index_maps().first(b_rank);
  // For each `single_input_dimension` map with an explicit bound in `b`, mark
  // the corresponding bound in `a` as explicit.
  for (DimensionIndex b_dim = 0; b_dim < b_rank; ++b_dim) {
    auto& map = maps[b_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension ||
        map.stride() == 0) {
      continue;
    }
    const DimensionIndex a_dim = map.input_dimension();
    assert(a_dim >= 0 && a_dim < a_rank);
    bool implicit_lower = b_implicit_lower_bounds[b_dim];
    bool implicit_upper = b_implicit_upper_bounds[b_dim];
    if (map.stride() < 0) {
      std::swap(implicit_lower, implicit_upper);
    }
    if (!implicit_lower) a_implicit_lower_bounds[a_dim] = false;
    if (!implicit_upper) a_implicit_upper_bounds[a_dim] = false;
  }
}

}  // namespace

absl::Status PropagateBounds(BoxView<> b, DimensionSet b_implicit_lower_bounds,
                             DimensionSet b_implicit_upper_bounds,
                             TransformRep* a_to_b, MutableBoxView<> a) {
  auto status = PropagateBoundsImpl(b, b_implicit_lower_bounds,
                                    b_implicit_upper_bounds, a_to_b, a);
  if (!status.ok()) {
    // Augment failed calls with transform and domain
    std::ostringstream os;
    internal_index_space::PrintToOstream(os, a_to_b);
    std::string str = os.str();
    absl::StrReplaceAll({{"\n", " "}}, &str);

    AddStatusPayload(status, "transform", absl::Cord(str));
    AddStatusPayload(status, "domain", absl::Cord(tensorstore::StrCat(b)));
  }
  return status;
}

absl::Status PropagateExplicitBounds(BoxView<> b, TransformRep* a_to_b,
                                     MutableBoxView<> a) {
  return PropagateBounds(b, false, false, a_to_b, a);
}

absl::Status PropagateBounds(BoxView<> b, DimensionSet b_implicit_lower_bounds,
                             DimensionSet b_implicit_upper_bounds,
                             TransformRep* a_to_b, MutableBoxView<> a,
                             DimensionSet& a_implicit_lower_bounds,
                             DimensionSet& a_implicit_upper_bounds) {
  PropagateImplicitBoundState(b.rank(), b_implicit_lower_bounds,
                              b_implicit_upper_bounds, a_to_b, a.rank(),
                              a_implicit_lower_bounds, a_implicit_upper_bounds);
  return PropagateBounds(b, b_implicit_lower_bounds, b_implicit_upper_bounds,
                         a_to_b, a);
}

Result<TransformRep::Ptr<>> PropagateBoundsToTransform(
    BoxView<> b_domain, DimensionSet b_implicit_lower_bounds,
    DimensionSet b_implicit_upper_bounds, TransformRep::Ptr<> a_to_b) {
  const DimensionIndex b_rank = b_domain.rank();
  if (!a_to_b) {
    a_to_b = TransformRep::Allocate(b_rank, b_rank);
    a_to_b->input_rank = a_to_b->output_rank = b_rank;
    SetToIdentityTransform(a_to_b->output_index_maps().first(b_rank));
    a_to_b->input_domain(b_rank).DeepAssign(b_domain);
    a_to_b->implicit_lower_bounds = b_implicit_lower_bounds;
    a_to_b->implicit_upper_bounds = b_implicit_upper_bounds;
    internal_index_space::DebugCheckInvariants(a_to_b.get());
    return a_to_b;
  }
  const DimensionIndex a_rank = a_to_b->input_rank;
  Box<dynamic_rank(internal::kNumInlinedDims)> bounds_temp(a_rank);
  TENSORSTORE_RETURN_IF_ERROR(PropagateBounds(b_domain, b_implicit_lower_bounds,
                                              b_implicit_upper_bounds,
                                              a_to_b.get(), bounds_temp));
  a_to_b = MutableRep(std::move(a_to_b));
  a_to_b->input_domain(a_rank).DeepAssign(bounds_temp);
  PropagateImplicitBoundState(
      b_rank, b_implicit_lower_bounds, b_implicit_upper_bounds, a_to_b.get(),
      a_rank, a_to_b->implicit_lower_bounds, a_to_b->implicit_upper_bounds);
  const bool domain_is_explicitly_empty = IsDomainExplicitlyEmpty(a_to_b.get());
  const auto output_index_maps = a_to_b->output_index_maps().first(b_rank);
  for (DimensionIndex b_dim = 0; b_dim < b_rank; ++b_dim) {
    auto& map = output_index_maps[b_dim];
    if (map.method() != OutputIndexMethod::array) continue;
    if (domain_is_explicitly_empty) {
      map.SetConstant();
      map.offset() = 0;
      map.stride() = 0;
      continue;
    }
    auto& index_array_data = map.index_array_data();
    TENSORSTORE_ASSIGN_OR_RETURN(
        const IndexInterval propagated_bounds,
        GetAffineTransformDomain(
            OptionallyImplicitIndexInterval(b_domain[b_dim],
                                            b_implicit_lower_bounds[b_dim],
                                            b_implicit_upper_bounds[b_dim])
                .effective_interval(),
            map.offset(), map.stride()));
    index_array_data.index_range =
        Intersect(propagated_bounds, index_array_data.index_range);
  }
  internal_index_space::DebugCheckInvariants(a_to_b.get());
  return a_to_b;
}

Result<TransformRep::Ptr<>> PropagateExplicitBoundsToTransform(
    BoxView<> b_domain, TransformRep::Ptr<> a_to_b) {
  return PropagateBoundsToTransform(b_domain, false, false, std::move(a_to_b));
}

}  // namespace internal_index_space
}  // namespace tensorstore
