// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_IO_HANDLE_IMPL_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_IO_HANDLE_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"

namespace tensorstore {
namespace internal_ocdbt {

struct ReadCoalesceOptions {
  int64_t max_overhead_bytes_per_request;
  int64_t max_merged_bytes_per_request;
  absl::Duration max_interval;
};

/// Returns an `IoHandle` handle based on the specified arguments.
IoHandle::Ptr MakeIoHandle(
    const Context::Resource<tensorstore::internal::DataCopyConcurrencyResource>&
        data_copy_concurrency,
    internal::CachePool* cache_pool, const KvStore& base_kvstore,
    ConfigStatePtr config_state, size_t write_target_size = 0,
    std::optional<ReadCoalesceOptions> read_coalesce_options = std::nullopt);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_IO_HANDLE_IMPL_H_
