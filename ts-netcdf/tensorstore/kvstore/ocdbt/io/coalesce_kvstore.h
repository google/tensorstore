// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_COALESCE_KVSTORE_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_COALESCE_KVSTORE_H_

#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Adapts a base kvstore to coalesce read ranges.
///
/// Concurrent reads for the same key may be merged if the ranges are
/// separated by less than threshold bytes. 1MB may be a reasonable value
/// for reducing GCS reads in the OCDBT driver.
kvstore::DriverPtr MakeCoalesceKvStoreDriver(kvstore::DriverPtr base,
                                             size_t threshold,
                                             size_t merged_threshold,
                                             absl::Duration interval,
                                             Executor executor);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_INDIRECT_DATA_KVSTORE_DRIVER_H_
