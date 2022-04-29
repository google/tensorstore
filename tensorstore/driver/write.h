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

#ifndef TENSORSTORE_DRIVER_WRITE_H_
#define TENSORSTORE_DRIVER_WRITE_H_

#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/progress.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

/// Options for DriverWrite.
struct DriverWriteOptions {
  /// Callback to be invoked after each chunk is copied or committed.  Must
  /// remain valid until the returned `commit_future` becomes ready.  May be
  /// `nullptr` to indicate that progress information is not needed.  The
  /// callback may be invoked concurrently from multiple threads.  All
  /// WriteProgress values are monotonically increasing.  The `total_elements`
  /// value does not change after the first call.
  WriteProgressFunction progress_function;

  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  DataTypeConversionFlags data_type_conversion_flags =
      DataTypeConversionFlags::kSafeAndImplicit;
};

/// Copies data from an array to a TensorStore driver.
///
/// If an error occurs while writing, the `target` `TensorStore` may be left in
/// a partially-written state.
///
/// \param executor Executor to use for copying data.
/// \param source Source array.
/// \param target Target TensorStore.
/// \param options Specifies optional progress function.
/// \returns A WriteFutures object that can be used to monitor completion.
/// \error `absl::StatusCode::kInvalidArgument` if the domain of `source` cannot
///     be aligned to the resolved domain of `target.transform` via
///     `AlignDomainTo`.
/// \error `absl::StatusCode::kInvalidArgument` if `source.dtype()` cannot
///     be converted to `target.driver->dtype()`.
WriteFutures DriverWrite(Executor executor,
                         TransformedSharedArray<const void> source,
                         DriverHandle target, DriverWriteOptions options);

WriteFutures DriverWrite(TransformedSharedArray<const void> source,
                         DriverHandle target, WriteOptions options);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_WRITE_H_
