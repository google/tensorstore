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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_TRANSFORMED_ARRAY_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_TRANSFORMED_ARRAY_H_

#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Returns an `NDIterable` representation of `array`.
///
/// \param array The array to iterate over.  The data must remain valid as long
///     as the iterable is used.  A non-`Shared` array guaranteed to remain
///     valid for the lifetime of the returned `NDIterable` may be passed using
///     `UnownedToShared`.
/// \param arena Allocation arena to use, must remain valid until after the
///     returned `NDIterable` is destroyed.
/// \returns Non-null pointer to `NDIterable`.
/// \error `absl::StatusCode::kOutOfBounds` if any index arrays contain
///     out-of-bounds indices.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the iteration state.
Result<NDIterable::Ptr> GetTransformedArrayNDIterable(
    TransformedArray<Shared<const void>> array, Arena* arena);

/// Returns an `NDIterable` representation of `array | transform`.
///
/// \param array The array to iterate over.  The data must remain valid as long
///     as the iterable is used, but the layout need not remain valid after this
///     function returns.  A non-`Shared` array guaranteed to remain valid for
///     the lifetime of the returned `NDIterable` may be passed using
///     `UnownedToShared`.
/// \param transform Index transform to apply to `array`.  May be a null
///     transform to indicate an identity transform.
/// \param arena Allocation arena to use, must remain valid until after the
///     returned `NDIterable` is destroyed.
/// \returns Non-null pointer to `NDIterable`.
/// \error `absl::StatusCode::kOutOfBounds` if any index arrays contain
///     out-of-bounds indices.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the iteration state.
Result<NDIterable::Ptr> GetTransformedArrayNDIterable(
    SharedOffsetArrayView<const void> array, IndexTransformView<> transform,
    Arena* arena);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_TRANSFORMED_ARRAY_H_
