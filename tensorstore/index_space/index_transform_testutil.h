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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_TESTUTIL_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_TESTUTIL_H_

#include "absl/random/bit_gen_ref.h"
#include "tensorstore/index_space/index_transform.h"

namespace tensorstore {
namespace internal {

/// Applies a random dim expression to `transform`.
IndexTransform<> ApplyRandomDimExpression(absl::BitGenRef gen,
                                          IndexTransform<> transform);

/// Generates a random index transform whose range is within the specified
/// output bounds.
///
/// This starts with an identity transform over `output_bounds` and then mutates
/// it by calling `ApplyRandomDimExpression` `num_ops` times.
///
/// \param gen Random bit generator to use.
/// \param output_bounds Constraint on output range.
/// \param num_ops Number of times to call `ApplyRandomDimExpression`.
IndexTransform<> MakeRandomIndexTransform(absl::BitGenRef gen,
                                          BoxView<> output_bounds,
                                          size_t num_ops);

struct MakeStridedIndexTransformForOutputSpaceParameters {
  /// Maximum number of dummy input dimensions that may be added.
  DimensionIndex max_new_dims = 1;

  /// Specifies whether dummy input dimensions have a domain of `[0, 1)` rather
  /// than `(-inf, +inf)`.  If set to `false`, the returned transform is not
  /// actually invertible.
  bool new_dims_are_singleton = true;

  /// If `1`, the returned transform is invertible (has a stride of +/-1 for all
  /// output index maps) and the range is exactly `output_bounds`.  If greater
  /// than `1`, the returned transform is not necessarily invertible (may
  /// include output index maps with strides not equal to +/-1) and the output
  /// range may be only a subset of `output_bounds`.
  Index max_stride = 1;

  IndexInterval offset_interval = IndexInterval::UncheckedClosed(-4, 4);
};

/// Generates a random invertible/strided-only index transform whose range is
/// within the specified output bounds.
///
/// In the returned transform, all output index maps are
/// `single_input_dimension` with a unique input dimension.
///
/// \param gen Random bit generator to use.
/// \param output_domain Constraint on output domain.
/// \param p Other parameters.
IndexTransform<> MakeRandomStridedIndexTransformForOutputSpace(
    absl::BitGenRef gen, IndexDomainView<> output_domain,
    const MakeStridedIndexTransformForOutputSpaceParameters& p = {});

struct MakeStridedIndexTransformForInputSpaceParameters {
  /// Maximum number of dummy output dimensions that may be added.
  DimensionIndex max_new_dims = 1;

  /// If `1`, the returned transform is invertible (has a stride of +/-1 for all
  /// output index maps).  If greater than `1`, the returned transform is not
  /// necessarily invertible (may include output index maps with strides not
  /// equal to +/-1).
  Index max_stride = 1;

  IndexInterval offset_interval = IndexInterval::UncheckedClosed(-4, 4);
};

/// Generates a random invertible/strided-only index transform with the
/// specified domain.
///
/// In the returned transform, all input dimensions map to a unique output
/// dimension via a `single_input_dimension` map.
///
/// \param gen Random bit generator to use.
/// \param input_domain Input domain.
/// \param p Other parameters.
IndexTransform<> MakeRandomStridedIndexTransformForInputSpace(
    absl::BitGenRef gen, IndexDomainView<> input_domain,
    const MakeStridedIndexTransformForInputSpaceParameters& p = {});

/// Sets `permutation` to a random permutation of
/// `[0, 1, ..., permutation.size())`.
void MakeRandomDimensionOrder(absl::BitGenRef gen,
                              span<DimensionIndex> permutation);

struct MakeRandomBoxParameters {
  DimensionIndex min_rank = 0, max_rank = 3;
  IndexInterval origin_range = IndexInterval::UncheckedClosed(-10, 10);
  IndexInterval size_range = IndexInterval::UncheckedClosed(1, 5);
};

/// Generates a random box with the specified constraints.
Box<> MakeRandomBox(absl::BitGenRef gen, const MakeRandomBoxParameters& p = {});

/// Chooses a random box of the specified `shape` within `outer`.
Box<> ChooseRandomBoxPosition(absl::BitGenRef gen, BoxView<> outer,
                              span<const Index> shape);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_TESTUTIL_H_
