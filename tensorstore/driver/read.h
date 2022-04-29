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

#ifndef TENSORSTORE_DRIVER_READ_H_
#define TENSORSTORE_DRIVER_READ_H_

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal {

/// Options for DriverRead.
struct DriverReadOptions {
  /// Callback to be invoked after each chunk is completed.  Must remain valid
  /// until the returned future becomes ready.  May be `nullptr` to indicate
  /// that progress information is not needed.  The callback may be invoked
  /// concurrently from multiple threads.  All ReadProgress values are
  /// monotonically increasing.  The `total_elements` value does not change
  /// after the first call.
  ReadProgressFunction progress_function;

  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  DataTypeConversionFlags data_type_conversion_flags =
      DataTypeConversionFlags::kSafeAndImplicit;
};

struct DriverReadIntoNewOptions {
  /// Callback to be invoked after each chunk is completed.  Must remain valid
  /// until the returned future becomes ready.  May be `nullptr` to indicate
  /// that progress information is not needed.  The callback may be invoked
  /// concurrently from multiple threads.  All ReadProgress values are
  /// monotonically increasing.  The `total_elements` value does not change
  /// after the first call.
  ReadProgressFunction progress_function;
};

/// Copies data from a TensorStore driver to an array.
///
/// If an error occurs while reading, the `target` array may be left in a
/// partially-written state.
///
/// \param executor Executor to use for copying data.
/// \param source Source TensorStore.
/// \param target Destination array.
/// \param options Specifies optional progress function.
/// \returns A future that becomes ready when the data has been copied or an
///     error occurs.  The `target` array must remain valid until the returned
///     future becomes ready.
/// \error `absl::StatusCode::kInvalidArgument` if the resolved domain of
///     `source.transform` cannot be aligned to the domain of `target` via
///     `AlignDomainTo`.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->dtype()`
///     cannot be converted to `target.dtype()`.
Future<void> DriverRead(Executor executor, DriverHandle source,
                        TransformedSharedArray<void> target,
                        DriverReadOptions options);

Future<void> DriverRead(DriverHandle source,
                        TransformedSharedArray<void> target,
                        ReadOptions options);

/// Copies data from a TensorStore driver to a newly-allocated array.
///
/// \param executor Executor to use for copying data.
/// \param source Read source.
/// \param target_dtype Data type of newly-allocated destination array.
/// \param target_layout_order ChunkLayout order of newly-allocated destination
///     array.
/// \param options Specifies optional progress function.
/// \returns A future that becomes ready when the data has been copied or an
///     error occurs.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->dtype()`
///     cannot be converted to `target_dtype`.
Future<SharedOffsetArray<void>> DriverReadIntoNewArray(
    Executor executor, DriverHandle source, DataType target_dtype,
    ContiguousLayoutOrder target_layout_order,
    DriverReadIntoNewOptions options);

Future<SharedOffsetArray<void>> DriverReadIntoNewArray(
    DriverHandle source, ReadIntoNewArrayOptions options);

/// Copies `chunk` transformed by `chunk_transform` to `target`.
absl::Status CopyReadChunk(
    ReadChunk::Impl& chunk, IndexTransform<> chunk_transform,
    const DataTypeConversionLookupResult& chunk_conversion,
    TransformedArray<void, dynamic_rank, view> target);

absl::Status CopyReadChunk(ReadChunk::Impl& chunk,
                           IndexTransform<> chunk_transform,
                           TransformedArray<void, dynamic_rank, view> target);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_READ_H_
