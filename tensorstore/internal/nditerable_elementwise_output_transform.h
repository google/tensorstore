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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_OUTPUT_TRANSFORM_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_OUTPUT_TRANSFORM_H_

#include "tensorstore/data_type.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Returns a write-only NDIterable that applies the specified element-wise
/// closure and then copies to the specified `output`.  The elementwise closure
/// receives two arrays: the first array nis the input, and the second array
/// should be updated with the result.
///
/// This can be used for data type conversion among other things.
///
/// The returned iterable always requires an external buffer
/// (`GetIterationBufferConstraint` always returns an
/// `IterationBufferConstraint` with `external=true`).  The specified `closure`
/// is called once per block to fill the externally-supplied buffer.
///
/// \param output Writable output iterable to adapt.
/// \param input_data_type The input data type.
/// \param closure Closure that takes an input array of data type
///     `input_data_type`, an output array of data type `output.data_type()`,
///     fills the output array based on the input, and returns the number of
///     elements successfully written to the output array.
/// \param arena Arena that may be used for memory allocation.
NDIterable::Ptr GetElementwiseOutputTransformNDIterable(
    NDIterable::Ptr output, DataType input_data_type,
    ElementwiseClosure<2, Status*> closure, Arena* arena);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_ELEMENTWISE_OUTPUT_TRANSFORM_H_
