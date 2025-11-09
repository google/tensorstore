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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_H_

#include "tensorstore/index.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns a new transform with the input dimension order reversed.
///
/// If `transform` is `nullptr`, returns `nullptr`.
///
/// \param transform The existing transform.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
TransformRep::Ptr<> TransposeInputDimensions(TransformRep::Ptr<> transform,
                                             bool domain_only);

/// Returns a new transform with the input dimension order permuted.
///
/// If `transform` is `nullptr`, returns `nullptr`.
///
/// \param transform The existing transform.
/// \param permutation Specifies the old dimension index corresponding to each
///     new dimension: `permutation[i]` is the old dimension index corresponding
///     to new dimension `i`.
/// \param domain_only Indicates the output dimensions of `transform` should be
///     ignored, and returned transform should have an output rank of 0.
/// \dchecks If `transform != nullptr`, `permutation` is a valid permutation for
///     `transform->input_rank`.
TransformRep::Ptr<> TransposeInputDimensions(
    TransformRep::Ptr<> transform, span<const DimensionIndex> permutation,
    bool domain_only);

/// Returns a new transform with the output dimension order reversed.
///
/// If `transform` is `nullptr`, returns `nullptr`.
///
/// \param transform The existing transform.
TransformRep::Ptr<> TransposeOutputDimensions(TransformRep::Ptr<> transform);

/// Returns a new transform with the output dimension order permuted.
///
/// If `transform` is `nullptr`, returns `nullptr`.
///
/// \param transform The existing transform.
/// \param permutation Specifies the old dimension index corresponding to each
///     new dimension: `permutation[i]` is the old dimension index corresponding
///     to new dimension `i`.
/// \dchecks If `transform != nullptr`, `permutation` is a valid permutation for
///     `transform->output_rank`.
TransformRep::Ptr<> TransposeOutputDimensions(
    TransformRep::Ptr<> transform, span<const DimensionIndex> permutation);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSPOSE_H_
