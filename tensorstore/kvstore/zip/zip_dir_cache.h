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

#ifndef TENSORSTORE_KVSTORE_ZIP_ZIP_DIR_CACHE_H_
#define TENSORSTORE_KVSTORE_ZIP_ZIP_DIR_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>

#include "absl/base/optimization.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/zip/cached_dir.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_zip_kvstore {

// Cache used for reading the ZIP directory.
class ZipDirectoryCache : public internal::AsyncCache {
  using Base = internal::AsyncCache;

 public:
  using ReadData = CachedDir;

  explicit ZipDirectoryCache(kvstore::DriverPtr kvstore_driver,
                             Executor executor)
      : kvstore_driver_(std::move(kvstore_driver)),
        executor_(std::move(executor)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = ZipDirectoryCache;

    size_t ComputeReadDataSizeInBytes(const void* read_data) final;

    void DoRead(AsyncCacheReadRequest request) final;
  };

  Entry* DoAllocateEntry() final;
  size_t DoGetSizeofEntry() final;

  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    ABSL_UNREACHABLE();
  }

  kvstore::DriverPtr kvstore_driver_;
  Executor executor_;

  const Executor& executor() { return executor_; }
};

}  // namespace internal_zip_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZIP_ZIP_DIR_CACHE_H_
