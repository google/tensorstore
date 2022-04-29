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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_UTIL_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_UTIL_H_

#include <iosfwd>

#include "absl/container/inlined_vector.h"
#include "tensorstore/box.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

/// Result type for `PropagateIndexTransformDownsampling`.
struct PropagatedIndexTransformDownsampling {
  IndexTransform<> transform;
  /// Downsample factors for each input dimension of `transform`.
  absl::InlinedVector<Index, internal::kNumInlinedDims>
      input_downsample_factors;

  // Comparison and streaming operators are for testing only.

  friend bool operator==(const PropagatedIndexTransformDownsampling& a,
                         const PropagatedIndexTransformDownsampling& b) {
    return a.transform == b.transform &&
           a.input_downsample_factors == b.input_downsample_factors;
  }
  friend bool operator!=(const PropagatedIndexTransformDownsampling& a,
                         const PropagatedIndexTransformDownsampling& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(
      std::ostream& os, const PropagatedIndexTransformDownsampling& x);
};

/// Propagates a downsampling operation through an index transform.
///
/// Given a `downsampled_transform` from index space `downsampled_a` ->
/// `downsampled_b`, computes `propagated.transform` from index space `a` to
/// `b`, and `propagated.input_downsample_factors`, such that:
///
///   downsampling `b` by `output_downsample_factors` (with bounds of
///   `output_base_bounds`) and then transforming by `downsampled_transform`
///
/// is equivalent to:
///
///   transforming `b` by `propgated.transform` and then downsampling by
///   `propgated.input_downsample_factors` (and possibly "squeezing" some
///   singleton dimensions that were added).
///
/// Note that this function assumes downsampling is performed using a method
/// such as `DownsampleMethod::kMean` where an output value can be computed as
/// long as there is at least one in-bounds value.  This function is *not* for
/// use with `DownsampleMethod::kStride`.
///
/// This function supports all index transforms, but some case can be handled
/// more efficiently than others:
///
/// 1. For output dimensions `output_dim` for which
///    `output_downsample_factors[output_dim] == 1`, the resultant
///    `propagated.transform` just uses the same output index map unchanged.
///
/// 2. If output dimension `output_dim` has a `single_input_dimension` map with
///    stide of +/-1 from a unique `input_dim`, then
///    `propagated.input_downsample_factors[input_dim]` is set to
///    `output_downsample_factors[output_dim]` and `transform` uses the same
///    output index map, except that the output offset and input bounds for
///    `input_dim` are adjusted.
///
/// 3. If output dimension `output_dim` has a `constant `map, then it is
///    converted to a `single_input_dimension` map from an additional synthetic
///    input dimension added to `propagated.transform`.
///
/// 4. Otherwise, the output map for `output_dim` is converted to an index array
///    map that depends on an additional synthetic input dimension added to
///    `propagated.transform` with domain `[0, downsample_factor)`, where
///    `downsample_factor = output_downsample_factors[output_dim]`.  Note that
///    this case applies if the output index map in `downsampled_transform` is
///    an index array map or a `single_input_dimension` map with non-unit stride
///    or non-unique input dimension.  It is possible that some of the
///    `downsample_factor` indices within a given downsampling block are outside
///    `output_base_bounds` (this function returns an error if none are in
///    bounds, though).  In that case, since the index transform representation
///    does not support ragged arrays, any out of bound indices are clamped to
///    fit in `output_base_bounds`.  If the downsample factor is greater than 2,
///    this *does* affect the result of the average computation within these
///    boundary blocks.
///
/// If any synthetic input dimensions are added to `transform`, they are
/// guaranteed to become singleton dimensions after downsampling by
/// `propagated.input_downsample_factors`, and they must then be "squeezed"
/// (eliminated) to maintain the equivalence with downsampling before
/// transforming.
///
/// \param downsampled_transform The transform between the downsampled index
///     domains.
/// \param output_base_bounds Bounds on the range of the returned
///     `propagated.transform`.  The returned transform is guaranteed to have a
///     range contained in `output_base_bounds`.  If the downsampling block
///     corresponding to a position within the range of `downsampled_transform`
///     does not intersect `output_base_bounds`, an error is returned.  Note
///     that the downsampling block must intersect, but need not be fully
///     contained in, `output_base_bound`.
/// \param output_downsample_factors Factors by which to downsample each
///     dimension of `b`.
/// \param propagated[out] The propagated result.
/// \pre `downsampled_transform.valid()`
/// \dchecks `output_downsample_factors.size() == output_base_bounds.rank()`
/// \dchecks `output_base_bounds.rank() == downsampled_transform.output_rank()`
/// \dchecks `output_downsample_factors[i] > 0` for all `i`.
/// \error `absl::StatusCode::kOutOfRange` if downsampling would require data
///     outside of `output_base_bounds`.
absl::Status PropagateIndexTransformDownsampling(
    IndexTransformView<> downsampled_transform, BoxView<> output_base_bounds,
    span<const Index> output_downsample_factors,
    PropagatedIndexTransformDownsampling& propagated);

/// Same as above, but with `Result` return value.
Result<PropagatedIndexTransformDownsampling>
PropagateIndexTransformDownsampling(
    IndexTransformView<> downsampled_transform, BoxView<> output_base_bounds,
    span<const Index> output_downsample_factors);

/// Computes the maximum downsampled interval that can be computed from the
/// given base interval.
///
/// If `method == kStride`, a downsampled position `x` can be computed if
/// `base_interval` contains `x * downsample_factor`.
///
/// If `method == kMean`, a downsampled position `x` can be computed if
/// `base_interval` intersects
/// `[x * downsample_factor, (x + 1) * downsample_factor - 1]`.
IndexInterval DownsampleInterval(IndexInterval base_interval,
                                 Index downsample_factor,
                                 DownsampleMethod method);

/// Computes the maximum downsampled region that can be computed from the given
/// base region.
///
/// This simply calls `DownsampleInterval` for each dimension.
///
/// \param base_bounds The original (not downsampled) bounds.
/// \param downsampled_bounds[out] The downsampled bounds.
/// \param downsample_factors The downsample factors for each dimension.
/// \param downsample_method The method to use, determines rounding.
/// \dchecks `base_bounds.rank() == downsampled_bounds.rank()`
/// \dchecks `base_bounds.rank() == downsample_factors.size()`
void DownsampleBounds(BoxView<> base_bounds,
                      MutableBoxView<> downsampled_bounds,
                      span<const Index> downsample_factors,
                      DownsampleMethod method);

/// Downsamples `base_domain`.
///
/// The returned domain copies `labels`, `implicit_lower_bounds`, and
/// `implicit_upper_bounds` from `base_domain`.  The bounds are obtained from
/// calling `DownsampleBounds`.
///
/// \param base_domain The base domain to downsample.
/// \param downsample_factors Downsample factor for each dimension of
///     `base_domain`.  The size must match the rank of `base_domain`.  All
///     factors must be positive.
/// \param downsample_method The downsampling method to use.
/// \returns The downsampled domain.
IndexDomain<> DownsampleDomain(IndexDomainView<> base_domain,
                               span<const Index> downsample_factors,
                               DownsampleMethod method);

/// Returns an identity transform over the domain obtained by downsampling
/// `base_domain`.
///
/// Equivalent to:
///
///     IdentityTransform(DownsampleDomain(base_domain, downsample_factors,
///                                        method))
IndexTransform<> GetDownsampledDomainIdentityTransform(
    IndexDomainView<> base_domain, span<const Index> downsample_factors,
    DownsampleMethod method);

/// Returns `true` if the range of `base_cell_transform` can be downsampled
/// independently.
///
/// This is true if, and only if, the following conditions are satisfied:
///
/// 1. `base_cell_transform` can be inverted by `InverseTransform`.
///
/// 2. Each dimension of the domain of the inverse transform is aligned either
///    to `base_bounds` or to a `downsample_factors` block boundary.
bool CanDownsampleIndexTransform(IndexTransformView<> base_transform,
                                 BoxView<> base_bounds,
                                 span<const Index> downsample_factors);

}  // namespace internal_downsample
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_UTIL_H_
