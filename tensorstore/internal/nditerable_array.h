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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_H_

#include "tensorstore/array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable.h"

namespace tensorstore {
namespace internal {

/// Returns an NDIterable representation of `array`.
///
/// \param array The array to iterate over.  The data must remain valid as long
///     as the iterable is used, but the layout need not remain valid after this
///     function returns.  A non-`Shared` array guaranteed to remain valid for
///     the lifetime of the returned `NDIterable` may be passed using
///     `UnownedToShared`.
/// \param arena Allocation arena to use, must remain valid until after the
///     returned `NDIterable` is destroyed.
/// \returns Non-null pointer to `NDIterable`.
NDIterable::Ptr GetArrayNDIterable(SharedOffsetArrayView<const void> array,
                                   Arena* arena);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_H_
