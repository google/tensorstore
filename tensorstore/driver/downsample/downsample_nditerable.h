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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_NDITERABLE_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_NDITERABLE_H_

#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

/// Provides a downsampled view of an existing `NDIterable`.
///
/// Does not support `DownsampleMethod::kStride`.  Striding-based downsampling
/// should instead be performed via an index transform.
///
/// \param base The base iterable.
/// \param base_domain The index domain of `base`.
/// \param downsample_factors Downsample factors for each dimension of
///     `base_domain`.  Must all be positive.
/// \param downsample_method The downsampling method.
/// \param target_rank The rank of the returned downsampled view.  This must be
///     less than or equal to `base_domain.rank()`, and any remaining dimensions
///     must be singleton dimensions after downsampling.
/// \param arena Arena to use for memory allocation.
/// \returns The downsampled view.
/// \dchecks `downsample_method` must not be `kStride` and must be supported for
///     `base->dtype()`.
internal::NDIterable::Ptr DownsampleNDIterable(
    internal::NDIterable::Ptr base, BoxView<> base_domain,
    span<const Index> downsample_factors, DownsampleMethod downsample_method,
    DimensionIndex target_rank, internal::Arena* arena);

/// Returns `true` if the specified downsampling `method` is supported for the
/// given `dtype`.
///
/// Refer to the `DownsampleMethod` documentation for details of which data
/// types are supported.
bool IsDownsampleMethodSupported(DataType dtype, DownsampleMethod method);

/// Validates that `IsDownsampleMethodSupported(dtype, downsample_method)`
/// returns `true`.
///
/// \returns `absl::OkStatus()` if
///     `IsDownsampleMethodSupported(dtype, downsample_method) == true`.
/// \error `absl::StatusCode::kInvalidArgument` otherwise.
absl::Status ValidateDownsampleMethod(DataType dtype,
                                      DownsampleMethod downsample_method);

}  // namespace internal_downsample
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_NDITERABLE_H_
