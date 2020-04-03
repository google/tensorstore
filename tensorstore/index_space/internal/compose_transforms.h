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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_COMPOSE_TRANSFORMS_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_COMPOSE_TRANSFORMS_H_

#include "tensorstore/index_space/internal/transform_rep.h"

namespace tensorstore {
namespace internal_index_space {

/// Sets `a_to_c` to be the composition of `b_to_c` and `a_to_b`.
///
/// \param b_to_c[in] The transform from index space "b" to index space "c".
/// \param can_move_from_b_to_c Specifies whether `b_to_c` may be modified.
/// \param a_to_b[in] The transform from index space "a" to index space "b".
/// \param can_move_from_a_to_b Specifies whether `a_to_b` may be modified.
/// \param a_to_c[out] The transform to be set to the composition of `a_to_b`
///     and `b_to_c`.
/// \dchecks `b_to_c != nullptr && a_to_b != nullptr && a_to_c != nullptr`.
/// \dchecks `b_to_c->input_rank == a_to_b->output_rank`.
/// \dchecks `a_to_c->output_rank_capacity >= b_to_c->output_rank`.
/// \dchecks `a_to_c->input_rank_capacity >= a_to_b->input_rank`.
/// \returns A success `Status()` or error.
Status ComposeTransforms(TransformRep* b_to_c, bool can_move_from_b_to_c,
                         TransformRep* a_to_b, bool can_move_from_a_to_b,
                         TransformRep* a_to_c);

/// Computes the composition of `b_to_c` and `a_to_b`.
///
/// \param b_to_c The transform from index space "b" to index space "c".
/// \param can_move_from_b_to_c Specifies whether `b_to_c` may be modified.
/// \param a_to_b The transform from index space "a" to index space "b".
/// \param can_move_from_a_to_b Specifies whether `a_to_b` may be modified.
/// \dchecks `b_to_c != nullptr && a_to_b != nullptr`.
/// \dchecks `b_to_c->input_rank == a_to_b->output_rank`
/// \returns A non-null pointer to a newly allocated transform with `input_rank
///     = a_to_b->input_rank`, `output_rank = b_to_c->output_rank`.
Result<TransformRep::Ptr<>> ComposeTransforms(TransformRep* b_to_c,
                                              bool can_move_from_b_to_c,
                                              TransformRep* a_to_b,
                                              bool can_move_from_a_to_b);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_COMPOSE_TRANSFORMS_H_
