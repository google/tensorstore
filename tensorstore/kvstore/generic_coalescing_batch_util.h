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
    BatchReadEntry<DerivedDriver, /*ReadRequest=*/ByteRangeReadRequest,
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
    : public GenericCoalescingBatchReadEntryBase<DerivedDriver> {
  using Base = GenericCoalescingBatchReadEntryBase<DerivedDriver>;
  using BatchEntryKey = typename Base::BatchEntryKey;
  using Request = typename Base::Request;
  using Base::batch_entry_key;
  using Base::request_batch;

  using Base::Base;

  void Submit(Batch::Impl::Entry::Ptr self, Batch::View batch) final {
    if (request_batch.requests.empty()) return;
    this->driver().executor()(
        [self = internal::static_pointer_cast<GenericCoalescingBatchReadEntry>(
             std::move(self))]() mutable { ProcessBatch(std::move(self)); });
  }

  static void ProcessBatch(
      internal::IntrusivePtr<GenericCoalescingBatchReadEntry> self) {
    // PRECONDITION: Each request's byte_range satisfies the IsRange()
    // constraint. See `HandleBatchRequestByGenericByteRangeCoalescing`.
    ForEachCoalescedRequest<Request>(
        self->request_batch.requests,
        self->driver().GetBatchReadCoalescingOptions(),
        [&](OptionalByteRangeRequest coalesced_byte_range,
            tensorstore::span<Request> coalesced_requests) {
          auto current_range = coalesced_byte_range.AsByteRange();
          kvstore::ReadOptions options;
          options.generation_conditions =
              std::get<kvstore::ReadGenerationConditions>(
                  self->batch_entry_key);
          options.staleness_bound = self->request_batch.staleness_bound;
          options.byte_range = current_range;
          auto read_future = self->driver().ReadImpl(
              kvstore::Key(std::get<kvstore::Key>(self->batch_entry_key)),
              std::move(options));
          read_future.Force();
          std::move(read_future)
              .ExecuteWhenReady(WithExecutor(
                  self->driver().executor(),
                  [self, current_range, coalesced_requests](
                      ReadyFuture<kvstore::ReadResult> future) {
                    TENSORSTORE_ASSIGN_OR_RETURN(
                        auto&& read_result, future.result(),
                        internal_kvstore_batch::SetCommonResult(
                            coalesced_requests, _));
                    ResolveCoalescedRequests(current_range, coalesced_requests,
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
      typename Entry::Request{std::move(promise), options.byte_range});
  return std::move(future);
}

}  // namespace internal_kvstore_batch
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GENERIC_COALESCING_BATCH_UTIL_H_
