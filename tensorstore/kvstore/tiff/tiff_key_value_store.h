// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_

#include <memory>

#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace kvstore {
namespace tiff_kvstore {

/// Creates a TiffKeyValueStore driver instance using resolved resources.
///
/// This function assumes the provided resources (cache_pool_res, data_copy_res)
/// have already been resolved/bound using a Context.
Result<DriverPtr> GetTiffKeyValueStoreDriver(
    DriverPtr base_kvstore,  // Base driver (e.g., file, memory)
    std::string path,        // Path within the base driver
    const Context::Resource<internal::CachePoolResource>& cache_pool_res,
    const Context::Resource<internal::DataCopyConcurrencyResource>&
        data_copy_res,
    const internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>&
        dir_cache_entry);

}  // namespace tiff_kvstore
}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
