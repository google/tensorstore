// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GENERIC_COALESCING_BATCH_UTIL_H_
#define TENSORSTORE_KVSTORE_GENERIC_COALESCING_BATCH_UTIL_H_

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include "tensorstore/batch.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_kvstore_batch {

template <typename DerivedDriver>
using GenericCoalescingBatchReadEntryBase =
    BatchReadEntry<DerivedDriver, ReadRequest<>,
                   // BatchEntryKey members:
                   kvstore::Key, kvstore::ReadGenerationConditions>;

// Generic batch read implementation that simply coalesces requests to the same
// key with the same generation constraints, and then dispatches each coalesced
// request independently to the driver.
//
// This may be used by drivers to implement batch read support when no specific
// optimizations are possible.
//
// \tparam DerivedDriver The kvstore driver, must implement several additional
// methods:
//
//     - `Future<ReadResult> ReadImpl(Key, ReadOptions)` that performs a regular
//       non-batch read (`ReadOptions::batch` will always be `no_batch`).
//
//     - `CoalescingOptions GetBatchReadCoalescingOptions()` that returns the
//       coalescing options to use.
//
//     - `Executor executor()` that returns an executor to use for handling
//       batch read operations.
template <typename DerivedDriver>
struct GenericCoalescingBatchReadEntry
    : public GenericCoalescingBatchReadEntryBase<DerivedDriver>,
      public internal::AtomicReferenceCount<
          GenericCoalescingBatchReadEntry<DerivedDriver>> {
  using Base = GenericCoalescingBatchReadEntryBase<DerivedDriver>;
  using BatchEntryKey = typename Base::BatchEntryKey;
  using Request = typename Base::Request;
  using Base::batch_entry_key;
  using Base::request_batch;

  GenericCoalescingBatchReadEntry(BatchEntryKey&& batch_entry_key_)
      : GenericCoalescingBatchReadEntryBase<DerivedDriver>(
            std::move(batch_entry_key_)),
        // Create an initial reference count that is implicitly transferred to
        // `Submit`.
        internal::AtomicReferenceCount<
            GenericCoalescingBatchReadEntry<DerivedDriver>>(
            /*initial_ref_count=*/1) {}

  // Submit is responsible for destroying the entry when done.
  void Submit(Batch::View batch) final {
    if (request_batch.requests.empty()) return;
    this->driver().executor()([this] { this->ProcessBatch(); });
  }

  void ProcessBatch() {
    // Take ownership of the initial reference. A separate reference will be
    // held for each coalesced read such that the entry will be destroyed once
    // all individual coalesced reads complete.
    internal::IntrusivePtr<GenericCoalescingBatchReadEntry> self(
        this, internal::adopt_object_ref);
    ForEachCoalescedRequest<Request>(
        request_batch.requests, this->driver().GetBatchReadCoalescingOptions(),
        [&](ByteRange coalesced_byte_range, span<Request> coalesced_requests) {
          kvstore::ReadOptions options;
          options.generation_conditions =
              std::get<kvstore::ReadGenerationConditions>(batch_entry_key);
          options.staleness_bound = request_batch.staleness_bound;
          options.byte_range = coalesced_byte_range;
          auto read_future = this->driver().ReadImpl(
              kvstore::Key(std::get<kvstore::Key>(batch_entry_key)),
              std::move(options));
          read_future.Force();
          std::move(read_future)
              .ExecuteWhenReady(WithExecutor(
                  this->driver().executor(),
                  [self, coalesced_byte_range, coalesced_requests](
                      ReadyFuture<kvstore::ReadResult> future) {
                    TENSORSTORE_ASSIGN_OR_RETURN(
                        auto&& read_result, future.result(),
                        internal_kvstore_batch::SetCommonResult(
                            coalesced_requests, _));
                    ResolveCoalescedRequests(coalesced_byte_range,
                                             coalesced_requests,
                                             std::move(read_result));
                  }));
        });
  }
};

/// Handles a batch request by coalescing requests into a single combined
/// non-batch read request.
///
/// See `GenericCoalescingBatchReadEntry` for details.
template <typename DerivedDriver>
Future<kvstore::ReadResult> HandleBatchRequestByGenericByteRangeCoalescing(
    DerivedDriver& driver, kvstore::Key&& key, kvstore::ReadOptions&& options) {
  if (!options.batch || options.byte_range.IsFull() ||
      !options.byte_range.IsRange()) {
    return driver.ReadImpl(std::move(key), std::move(options));
  }
  auto [promise, future] = PromiseFuturePair<kvstore::ReadResult>::Make();
  using Entry = GenericCoalescingBatchReadEntry<DerivedDriver>;
  Entry::template MakeRequest<Entry>(
      driver, std::move(key), std::move(options.generation_conditions),
      options.batch, options.staleness_bound,
      typename Entry::Request{{std::move(promise), options.byte_range}});
  return std::move(future);
}

}  // namespace internal_kvstore_batch
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GENERIC_COALESCING_BATCH_UTIL_H_
