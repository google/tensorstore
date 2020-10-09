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

#ifndef TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_
#define TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_

/// \file
/// Implementation details of `kvs_backed_chunk_driver.h`.

#include <memory>
#include <string>

#include "tensorstore/box.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvs_backed_chunk_driver {

/// Validates that the resize operation specified by
/// `new_{inclusive_min,exclusive_max}` can be applied to `current_domaian`
/// subject to the constraints of `{inclusive_min,exclusive_max}_constraint` and
/// `{expand,shrink}_only`.
///
/// For each value in `{inclusive_min,exclusive_max}_constraint` that is not
/// `kImplicit`, the corresponding bound of `current_domain` must be equal.
///
/// \param new_inclusive_min The new inclusive min bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param new_exclusive_max The new exclusive max bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param inclusive_min_constraint The inclusive min constraint vector of
///     length `current_domain.rank()`.
/// \param exclusive_max_constraint The inclusive max constraint vector of
///     length `current_domain.rank()`.
/// \param expand_only If `true`, the bounds must not shrink.
/// \param shrink_only If `true`, the bounds must not expand.
/// \returns `OkStatus()` if the constraints are satisfied.
/// \error `absl::StatusCode::kFailedPrecondition` if the constraints are not
///     satisfied.
/// \dchecks `current_domain.rank() == new_inclusive_min.size()`
/// \dchecks `current_domain.rank() == new_exclusive_max.size()`
/// \dchecks `current_domain.rank() == inclusive_min_constraint.size()`
/// \dchecks `current_domain.rank() == exclusive_max_constraint.size()`
Status ValidateResizeConstraints(BoxView<> current_domain,
                                 span<const Index> new_inclusive_min,
                                 span<const Index> new_exclusive_max,
                                 span<const Index> inclusive_min_constraint,
                                 span<const Index> exclusive_max_constraint,
                                 bool expand_only, bool shrink_only);

/// Specifies how to resize a DataCache.
struct ResizeParameters {
  /// `new_inclusive_min[i]` and `new_exclusive_max[i]` specify the new lower
  /// and upper bounds for dimension `i`, or may be `kImplicit` to indicate
  /// that the existing value should be retained.
  std::vector<Index> new_inclusive_min;
  std::vector<Index> new_exclusive_max;

  /// If `inclusive_min_constraint[i]` or `exclusive_max_constraint[i]` is not
  /// `kImplicit`, the existing lower/upper bound must match it in order for
  /// the resize to succeed.
  std::vector<Index> inclusive_min_constraint;
  std::vector<Index> exclusive_max_constraint;

  /// Fail if any bounds would be reduced.
  bool expand_only;

  /// Fail if any bounds would be increased.
  bool shrink_only;
};

/// Propagates bounds from `new_metadata` for component `component_index` to
/// `transform`.
///
/// \param data_cache Non-null pointer to the data cache.
/// \param new_metadata Non-null pointer to the new metadata, of the type
///     expected by `data_cache`.
/// \param component_index The component index.
/// \param transform The existing transform.
/// \param options Resolve options.
Result<IndexTransform<>> ResolveBoundsFromMetadata(
    DataCache* data_cache, const void* new_metadata,
    std::size_t component_index, IndexTransform<> transform,
    ResolveBoundsOptions options);

/// Validates a resize request for consistency with `transform` and `metadata`.
///
/// \param data_cache Non-null pointer to the data cache.
/// \param metadata Non-null pointer to the existing metadata, of the type
///     expected by `data_cache`.
/// \param component_index The component index.
/// \param transform The existing transform.
/// \param inclusive_min The new inclusive min bounds for the input domain of
///     `transform`, or `kImplicit` for no change.
/// \param exclusive_min The new exclusive max bounds for the input domain of
///     `transform`, or `kImplicit` for no change.
/// \param options The resize options.
/// \param transaction_mode The transaction mode.  If equal to
///     `atomic_isolated`, additional constraints are included in
///     `inclusive_min_constraint` and `exclusive_max_constraint` to ensure
///     consistency.
/// \returns The computed resize parameters for the output index space if the
///     resize request is valid.
/// \error `absl::StatusCode::kAborted` if the resize would be a no-op.
/// \error `absl::StatusCode::kFailedPrecondition` if the resize is not
///     compatible with `metadata`.
/// \error `absl::StatusCode::kInvalidArgument` if the resize is invalid
///     irrespective of `metadata`.
/// \remark Even in the case this function returns successfully, the request may
///     fail later due to concurrent modification of the stored metadata.
Result<ResizeParameters> GetResizeParameters(
    DataCache* data_cache, const void* metadata, size_t component_index,
    IndexTransformView<> transform, span<const Index> inclusive_min,
    span<const Index> exclusive_max, ResizeOptions options,
    TransactionMode transaction_mode);

}  // namespace internal_kvs_backed_chunk_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_
