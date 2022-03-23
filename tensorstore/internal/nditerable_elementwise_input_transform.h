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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_TRANSFORM_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_TRANSFORM_H_

#include <array>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Returns a read-only NDIterable that applies the specified element-wise
/// closure to the specified inputs.  The elementwise closure receives `Arity`
/// arrays: the first `Arity-1` arrays are the inputs and the last array should
/// be updated with the result.
///
/// This can be used for data type conversion among other things.
///
/// The returned iterable always requires an external buffer
/// (`GetIterationBufferConstraint` always returns an
/// `IterationBufferConstraint` with `external=true`).  The specified `closure`
/// is called once per block to fill the externally-supplied buffer.
///
/// \tparam Arity The arity of the elementwise closure, equal to one plus the
///     number of inputs.
/// \param inputs The input iterables.
/// \param output_dtype The output data type expected by `closure`.
/// \param closure The elementwise function.
/// \param arena Arena that may be used for memory allocation.
template <std::size_t Arity>
NDIterable::Ptr GetElementwiseInputTransformNDIterable(
    std::array<NDIterable::Ptr, Arity - 1> inputs, DataType output_dtype,
    ElementwiseClosure<Arity, absl::Status*> closure, Arena* arena);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_TRANSFORM_H_
