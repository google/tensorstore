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
#include "tensorstore/internal/cache/async_initialized_cache_mixin.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_tiff_kvstore {

// First attempt reads this many bytes.
inline constexpr std::size_t kInitialReadBytes = 1024;

struct TiffParseResult {
  bool full_read = false;  // Indicates if the entire file was read

  // Store the endian order for the TIFF file
  Endian endian;

  // Store all IFD directories in the TIFF file
  std::vector<TiffDirectory> directories;

  // Store all parsed image directories
  std::vector<ImageDirectory> image_directories;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.full_read, x.endian, x.directories, x.image_directories);
  };
};

class TiffDirectoryCache : public internal::AsyncCache,
                           public internal::AsyncInitializedCacheMixin {
  using Base = internal::AsyncCache;

 public:
  using ReadData = TiffParseResult;

  explicit TiffDirectoryCache(kvstore::DriverPtr kv, Executor exec)
      : kvstore_driver_(std::move(kv)), executor_(std::move(exec)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = TiffDirectoryCache;
    size_t ComputeReadDataSizeInBytes(const void* read_data) final;
    void DoRead(AsyncCacheReadRequest request) final;

    // Load external arrays identified during IFD parsing
    Future<void> LoadExternalArrays(
        std::shared_ptr<TiffParseResult> parse_result,
        tensorstore::TimestampedStorageGeneration stamp);

    absl::Status AnnotateError(const absl::Status& error, bool reading) {
      return GetOwningCache(*this).kvstore_driver_->AnnotateError(
          this->key(), reading ? "reading" : "writing", error);
    }
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

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_tiff_kvstore::TiffDirectoryCache::Entry)

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_DIR_CACHE_H_