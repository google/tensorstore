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

struct MakeRandomBoxParameters {
  DimensionIndex min_rank = 0, max_rank = 3;
  IndexInterval origin_range = IndexInterval::UncheckedClosed(-10, 10);
  IndexInterval size_range = IndexInterval::UncheckedClosed(1, 5);
};

/// Generates a random box with the specified constraints.
Box<> MakeRandomBox(absl::BitGenRef gen, const MakeRandomBoxParameters& p = {});

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_TESTUTIL_H_
