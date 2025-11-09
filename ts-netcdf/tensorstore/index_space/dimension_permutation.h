// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_INDEX_SPACE_DIMENSION_PERMUTATION_H_
#define TENSORSTORE_INDEX_SPACE_DIMENSION_PERMUTATION_H_

#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Transforms a dimension order for the output space of `transform` to a
/// corresponding dimension order for the input space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `OutputIndexMethod::single_input_dimension` output
/// index maps, then `input_perm` is simply mapped from `output_perm` according
/// to this correspondence, and `TransformInputDimensionOrder` computes the
/// inverse.
///
/// More generally, a dimension ``input_dim`` in `input_perm` is ordered
/// ascending by the first index ``j`` for which the output dimension
/// ``output_perm[j]`` maps to ``input_dim`` via a
/// `OutputIndexMethod::single_input_dimension` output index map, and then by
/// dimension index.  Input dimensions with no corresponding output dimension
/// are ordered last.
///
/// \param transform The index transform.
/// \param output_perm Permutation of
///     ``{0, 1, ..., transform.output_rank()-1}``.
/// \param input_perm[out] Pointer to array of length `transform.input_rank()`.
/// \relates ChunkLayout
void TransformOutputDimensionOrder(IndexTransformView<> transform,
                                   span<const DimensionIndex> output_perm,
                                   span<DimensionIndex> input_perm);

/// Transforms a dimension order for the input space of `transform` to a
/// corresponding dimension order for the output space of `transform`.
///
/// If there is a one-to-one onto correspondence between output dimensions and
/// input dimensions via `OutputIndexMethod::single_input_dimension` output
/// index maps, then `output_perm` is simply mapped from `input_perm` according
/// to this correspondence, and `TransformOutputDimensionOrder` computes the
/// inverse.
///
/// More generally, each output dimension ``output_dim`` mapped with a
/// `OutputIndexMethod::single_input_dimension` map is ordered ascending by
/// ``inv(input_perm)[output_dim]``, and then by dimension index.  Output
/// dimensions without a `OutputIndexMethod::single_input_dimension` map are
/// ordered last, and then by dimension index.
///
/// \param transform The index transform.
/// \param input_perm Permutation of ``{0, 1, ..., transform.input_rank()-1}``.
/// \param output_perm[out] Pointer to array of length
///     `transform.output_rank()`.
/// \relates ChunkLayout
void TransformInputDimensionOrder(IndexTransformView<> transform,
                                  span<const DimensionIndex> input_perm,
                                  span<DimensionIndex> output_perm);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_DIMENSION_PERMUTATION_H_
