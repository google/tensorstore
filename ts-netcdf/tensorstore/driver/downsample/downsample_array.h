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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_ARRAY_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_ARRAY_H_

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_downsample {

/// Downsamples `source` and stores the result in `target`.
///
/// \param source The source array to downsample.
/// \param target The target array, the bounds must match the downsampled bounds
///     of `source`.
/// \param downsample_factors The downsample factors for each dimension of
///     `source`.
/// \param method Downsampling method to use.
/// \error `absl::StatusCode::kInvalidArgument` if the bounds of `target` do not
///     equal the downsampled bounds of `source`.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `source.dtype() != target.dtype()`.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `source.rank() != downsample_factors.size()`.
absl::Status DownsampleArray(OffsetArrayView<const void> source,
                             OffsetArrayView<void> target,
                             span<const Index> downsample_factors,
                             DownsampleMethod method);

/// Same as above, but allocates a new result array of the appropriate size.
Result<SharedOffsetArray<void>> DownsampleArray(
    OffsetArrayView<const void> source, span<const Index> downsample_factors,
    DownsampleMethod method);

/// Same as `DownsampleArray`, but operates on `TransformedArrayView`
/// instead.
absl::Status DownsampleTransformedArray(TransformedArrayView<const void> source,
                                        TransformedArrayView<void> target,
                                        span<const Index> downsample_factors,
                                        DownsampleMethod method);

/// Same as above, but allocates a new result array of the appropriate size.
Result<SharedOffsetArray<void>> DownsampleTransformedArray(
    TransformedArrayView<const void> source,
    span<const Index> downsample_factors, DownsampleMethod method);

}  // namespace internal_downsample
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_ARRAY_H_
