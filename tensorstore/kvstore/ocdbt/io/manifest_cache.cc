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

#include "tensorstore/kvstore/ocdbt/io/manifest_cache.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_variant.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

auto& manifest_updates = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/manifest_updates",
    "OCDBT driver manifest updates");

auto& manifest_update_errors = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/manifest_update_errors",
    "OCDBT driver manifest update errors (typically retried)");

}  // namespace

std::size_t ManifestCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
}

void ManifestCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                    DecodeReceiver receiver) {
  GetOwningCache(*this).executor()(
      [value = std::move(value), receiver = std::move(receiver)]() mutable {
        std::shared_ptr<ReadData> read_data;
        if (value) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto manifest, DecodeManifest(*value),
              static_cast<void>(execution::set_error(receiver, _)));
          read_data = std::make_shared<Manifest>(std::move(manifest));
        }
        execution::set_value(receiver, std::move(read_data));
      });
}

void ManifestCache::Entry::DoEncode(std::shared_ptr<const ReadData> read_data,
                                    EncodeReceiver receiver) {
  GetOwningCache(*this).executor()([read_data = std::move(read_data),
                                    receiver = std::move(receiver)]() mutable {
    std::optional<absl::Cord> encoded;
    if (read_data) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          encoded, EncodeManifest(*read_data),
          static_cast<void>(execution::set_error(receiver, _)));
    }
    execution::set_value(receiver, std::move(encoded));
  });
}

ManifestCache::Entry* ManifestCache::DoAllocateEntry() { return new Entry; }

std::size_t ManifestCache::DoGetSizeofEntry() { return sizeof(Entry); }

ManifestCache::TransactionNode* ManifestCache::DoAllocateTransactionNode(
    AsyncCache::Entry& entry) {
  return new TransactionNode(static_cast<ManifestCache::Entry&>(entry));
}

void ManifestCache::TransactionNode::DoApply(ApplyOptions options,
                                             ApplyReceiver receiver) {
  auto read_future = Read(options.staleness_bound);
  read_future.Force();
  read_future.ExecuteWhenReady([receiver = std::move(receiver),
                                this](ReadyFuture<const void> future) mutable {
    auto& r = future.result();
    if (!r.ok()) {
      execution::set_error(receiver, r.status());
      return;
    }
    TimestampedStorageGeneration existing_stamp;
    std::shared_ptr<const Manifest> existing_manifest;
    {
      ReadLock<Manifest> lock(*this);
      existing_stamp = lock.stamp();
      existing_manifest = lock.shared_data();
    }
    auto update_future = update_function(existing_manifest);
    update_future.Force();
    update_future.ExecuteWhenReady(
        [receiver = std::move(receiver),
         existing_manifest = std::move(existing_manifest),
         existing_stamp = std::move(existing_stamp)](
            ReadyFuture<std::shared_ptr<const Manifest>> future) mutable {
          auto& r = future.result();
          if (!r.ok()) {
            manifest_update_errors.Increment();
            execution::set_error(receiver, r.status());
            return;
          }
          ReadState new_read_state;
          new_read_state.stamp = std::move(existing_stamp);
          new_read_state.data = std::move(*r);
          if (new_read_state.data != existing_manifest) {
            new_read_state.stamp.generation.MarkDirty();
          }
          manifest_updates.Increment();
          execution::set_value(receiver, std::move(new_read_state));
        });
  });
}

Future<const ManifestWithTime> ManifestCache::Entry::Update(
    UpdateFunction update_function) {
  Transaction transaction(TransactionMode::isolated);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(transaction));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transaction_node,
      GetWriteLockedTransactionNode(*this, open_transaction));
  transaction_node->update_function = std::move(update_function);
  auto [promise, future] = PromiseFuturePair<ManifestWithTime>::Make();
  transaction_node->promise = promise;
  LinkError(std::move(promise), transaction.future());
  static_cast<void>(transaction.CommitAsync());
  return future;
}

void ManifestCache::TransactionNode::WritebackSuccess(ReadState&& read_state) {
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG) << "WritebackSuccess";
  ManifestWithTime manifest_with_time{
      std::static_pointer_cast<const Manifest>(read_state.data),
      read_state.stamp.time};
  auto promise = std::move(this->promise);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
  // Mark `promise` as ready only after adding the new manifest to the cache.
  // Otherwise operations triggered from `promise` becoming ready may read the
  // old manifest.
  promise.SetResult(std::move(manifest_with_time));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
