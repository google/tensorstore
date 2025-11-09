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

#ifndef TENSORSTORE_INDEX_SPACE_TRANSFORM_BROADCASTABLE_ARRAY_H_
#define TENSORSTORE_INDEX_SPACE_TRANSFORM_BROADCASTABLE_ARRAY_H_

/// \file
///
/// Facilities for transforming arrays with broadcasting.
///
/// This is used to transform fill values.

#include "tensorstore/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Transforms `output_array`, which must be broadcast-compatible with
/// `output_domain` (corresponding to the output space of `transform`) to a
/// corresponding input array that is broadcast-compatible with the input domain
/// of `transform`.
///
/// \param transform The index transform.  Due to broadcasting, the result is
///     invariant to any translation that `transform` applies.
/// \param output_array Output array, must be broadcast-compatible with
///     `output_domain`.
/// \param output_domain Output domain to which `output_array` is broadcast
///     compatible.  May be a null (i.e. default constructed) `IndexDomain`, in
///     order to indicate that the `output_domain` is unspecified.  If
///     `output_domain` is unspecified, it is inferred from `transform`, and
///     `output_array` is subject to additional restrictions in order to ensure
///     the result is invariant to any possible choice of `output_domain`.
/// \dchecks `transform.valid()`
/// \returns The corresponding array for the input domain.
Result<SharedArray<const void>> TransformOutputBroadcastableArray(
    IndexTransformView<> transform, SharedArrayView<const void> output_array,
    IndexDomainView<> output_domain);

/// Transforms `input_array`, which must be broadcast-compatible with
/// `transform.domain()`, to a corresponding output array that is
/// broadcast-compatible with the output space of `transform`.
///
/// This is the inverse of `TransformOutputBroadcastableArray`.
///
/// \param transform The index transform.  Due to broadcasting, the result is
///     invariant to any translation that `transform` applies.
/// \param input_array Must be broadcast-compatible with `transform.domain()`.
/// \dchecks `transform.valid()`
/// \returns The corresponding array for the output space.
Result<SharedArray<const void>> TransformInputBroadcastableArray(
    IndexTransformView<> transform, SharedArrayView<const void> input_array);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_TRANSFORM_BROADCASTABLE_ARRAY_H_
