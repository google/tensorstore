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

#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_zip_kvstore {

struct Directory {
  struct Entry {
    // Filename within the zip file.
    std::string filename;

    // Zip central directory parameters.
    uint32_t crc;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t local_header_offset;
    uint64_t estimated_size;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.filename, x.crc, x.compressed_size, x.uncompressed_size,
               x.local_header_offset, x.estimated_size);
    };

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Entry& entry) {
      absl::Format(
          &sink,
          "Entry{filename=%s, crc=%d, compressed_size=%d, "
          "uncompressed_size=%d, local_header_offset=%d, estimated_size=%d}",
          entry.filename, entry.crc, entry.compressed_size,
          entry.uncompressed_size, entry.local_header_offset,
          entry.estimated_size);
    }
  };

  // Indicates whether the ZIP kvstore should issue reads for the entire file;
  // this is done when the initial read returns a range error.
  std::vector<Entry> entries;
  bool full_read;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.entries, x.full_read);
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Directory& entry) {
    absl::Format(&sink, "Directory{\n");
    for (const auto& entry : entry.entries) {
      absl::Format(&sink, "%v\n", entry);
    }
    absl::Format(&sink, "}");
  }
};

/// Cache used for reading the ZIP directory.
class ZipDirectoryCache : public internal::AsyncCache {
  using Base = internal::AsyncCache;

 public:
  using ReadData = Directory;

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
