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

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_DIR_CACHE_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_DIR_CACHE_H_

#include <stddef.h>

#include "absl/strings/cord.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_tiff_kvstore {

// First attempt reads this many bytes.
inline constexpr std::size_t kInitialReadBytes = 1024;

struct TiffDirectoryParseResult {
  // For step-1 this just captures the raw bytes we read.
  absl::Cord raw_data;
  bool full_read = false;  // identical meaning to zip cache.
};

class TiffDirectoryCache : public internal::AsyncCache {  
  using Base = internal::AsyncCache;
 public:
  using ReadData = TiffDirectoryParseResult;

  explicit TiffDirectoryCache(kvstore::DriverPtr kv, Executor exec)
      : kvstore_driver_(std::move(kv)), executor_(std::move(exec)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = TiffDirectoryCache;
    size_t ComputeReadDataSizeInBytes(const void* read_data) final;
    void DoRead(AsyncCacheReadRequest request) final;
  };

  Entry* DoAllocateEntry() final;
  size_t DoGetSizeofEntry() final;

  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    ABSL_UNREACHABLE();  // Not implemented for step-1
    return nullptr;
  }

  kvstore::DriverPtr kvstore_driver_;
  Executor executor_;

  const Executor& executor() { return executor_; }
};

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_DIR_CACHE_H_