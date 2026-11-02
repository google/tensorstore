// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_ZIP_ZIP_WRITE_CACHE_H_
#define TENSORSTORE_KVSTORE_ZIP_ZIP_WRITE_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_zip_kvstore {

struct ZipEntryState {
  std::string filename;
  internal_zip::ZipCompression compression_method =
      internal_zip::ZipCompression::kStore;

  absl::Cord data;
  bool is_compressed = false;

  uint32_t crc32 = 0;
  uint64_t uncompressed_size = 0;

  absl::Time mtime;
  absl::Time ctime;
  absl::Time atime;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.filename, x.compression_method, x.data, x.is_compressed, x.crc32,
             x.uncompressed_size, x.mtime, x.ctime, x.atime);
  };
};

struct ZipEntries {
  // Must be sorted by filename. This invariant is established in
  // HandleDecode (after parsing) and MergeForWriteback (after
  // merging mutations), and is required by TransactionNode::Read
  // for binary search.
  std::vector<ZipEntryState> entries;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.entries);
  };
};

class ZipWriteCache
    : public internal::KvsBackedCache<ZipWriteCache, internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ZipWriteCache, internal::AsyncCache>;

 public:
  using ReadData = ZipEntries;

  explicit ZipWriteCache(kvstore::DriverPtr base_kvstore_driver,
                         std::string base_kvstore_path, Executor executor,
                         internal_zip::ZipCompression compression =
                             internal_zip::ZipCompression::kDeflate)
      : Base(std::move(base_kvstore_driver)),
        base_kvstore_path_(std::move(base_kvstore_path)),
        executor_(std::move(executor)),
        compression_(compression) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = ZipWriteCache;

    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return internal::EstimateHeapUsage(*static_cast<const ReadData*>(data));
    }

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override;

    void DoEncode(EncodeOptions options, std::shared_ptr<const ZipEntries> data,
                  EncodeReceiver receiver) override;

    std::string GetKeyValueStoreKey() override {
      return GetOwningCache(*this).base_kvstore_path();
    }
  };

  class TransactionNode : public Base::TransactionNode,
                          public internal_kvstore::AtomicMultiPhaseMutation {
   public:
    using OwningCache = ZipWriteCache;
    using Base::TransactionNode::TransactionNode;

    absl::Mutex& mutex() override { return this->mutex_; }

    void PhaseCommitDone(size_t next_phase) override {}

    internal::TransactionState::Node& GetTransactionNode() override {
      return *this;
    }

    void Abort() override {
      this->AbortRemainingPhases();
      Base::TransactionNode::Abort();
    }

    std::string DescribeKey(std::string_view key) override;

    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;

    void StartApply();

    void AllEntriesDone(
        internal_kvstore::SinglePhaseMutation& single_phase_mutation) override;

    void MergeForWriteback(bool conditional);

    void RecordEntryWritebackError(
        internal_kvstore::ReadModifyWriteEntry& entry,
        absl::Status error) override {
      absl::MutexLock lock(mutex_);
      if (apply_status_.ok()) {
        apply_status_ = std::move(error);
      }
    }

    void Revoke() override {
      Base::TransactionNode::Revoke();
      {
        // Fence to wait for any concurrent mutation to finish.
        lock();
        unlock();
      }
      this->RevokeAllEntries();
    }

    void WritebackSuccess(ReadState&& read_state) override;
    void WritebackError() override;

    void InvalidateReadState() override;
    bool MultiPhaseReadsCommitted() override { return this->reads_committed_; }

    void Read(std::string_view key,
              ReadModifyWriteTarget::ReadModifyWriteReadOptions&& options,
              ReadModifyWriteTarget::ReadReceiver&& receiver) override;

    void ListUnderlying(kvstore::ListOptions options,
                        kvstore::ListReceiver receiver) override;

    void ListFromEntries(const ZipEntries& entries,
                         kvstore::ListOptions options,
                         kvstore::ListReceiver receiver);

    ApplyReceiver apply_receiver_;
    ApplyOptions apply_options_;
    absl::Status apply_status_;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  const std::string& base_kvstore_path() const { return base_kvstore_path_; }
  const Executor& executor() const { return executor_; }
  internal_zip::ZipCompression compression() const { return compression_; }

  // Called on successful writeback to invalidate the read-path
  // directory cache, avoiding one pull-based aborted retry
  // round-trip.
  using InvalidateCallback = std::function<void(absl::Time)>;
  void set_invalidate_callback(InvalidateCallback cb) {
    invalidate_callback_ = std::move(cb);
  }

 private:
  std::string base_kvstore_path_;
  Executor executor_;
  internal_zip::ZipCompression compression_;
  InvalidateCallback invalidate_callback_;
};

}  // namespace internal_zip_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ZIP_ZIP_WRITE_CACHE_H_
