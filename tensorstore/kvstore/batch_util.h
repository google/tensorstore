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

#ifndef TENSORSTORE_KVSTORE_BATCH_UTIL_H_
#define TENSORSTORE_KVSTORE_BATCH_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/batch_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

// This file defines utilities for use by kvstore drivers to implementing batch
// read support.
//
// Refer to the documentation of `BatchReadEntry` for details.

namespace tensorstore {
namespace internal_kvstore_batch {

// Common portion of read request used with `BatchReadEntry`.
//
// This is combined in `ReadRequest` with additional fields only used for some
// kvstore implementations.
struct ByteRangeReadRequest {
  Promise<kvstore::ReadResult> promise;
  OptionalByteRangeRequest byte_range;
};

// Individual read request (entry in batch) used with `BatchReadEntry`.
//
// Possibly `Member` types include:
//
// - `kvstore::ReadGenerationConditions`
// - `kvstore::Key`
// - some derived key type, like `uint64_t`.
template <typename... Member>
using ReadRequest = std::tuple<ByteRangeReadRequest, Member...>;

// Batch of read requests with an aggregate staleness bound, used by
// `BatchReadEntry`.
//
// The aggregate staleness bound is set to the maximum staleness bound of all
// individual requests.
template <typename RequestType>
struct RequestBatch {
  using Request = RequestType;
  absl::Time staleness_bound = absl::InfinitePast();
  absl::InlinedVector<Request, 1> requests;

  void AddRequest(absl::Time staleness_bound, Request&& request) {
    this->staleness_bound = std::max(this->staleness_bound, staleness_bound);
    requests.push_back(std::move(request));
  }
};

/// Parameter type used for constructing a batch entry key member of type `T`.
template <typename T>
using KeyConstructType =
    std::conditional_t<std::is_trivially_copyable_v<T>, T, T&&>;

/// Parameter type used for querying a batch entry key member of type `T`.
template <typename T>
using KeyParamType =
    std::conditional_t<std::is_trivially_copyable_v<T>, T, const T&>;

// Batch entry implementation that may be used by kvstore drivers to implement
// batch read support.
//
// This simply aggregates batch requests, grouped by the members that are
// included in the batch entry key, until the batch is submitted.
//
// Note: Neither the base `Batch::Impl::Entry` class nor this derived class use
// reference counting. However, derived classes, which are responsible for
// implementing `Submit`, may want to use reference counting if the `Submit`
// implementation makes multiple concurrent requests (for separately-coalesced
// groups), in order to extend the lifetime of the entry until all requests
// complete.
//
// \tparam DerivedDriver Derived kvstore driver type.
// \tparam RequestType Must be `ReadRequest<Member...>`.
// \tparam BatchEntryKeyMember Additional types to include in the
//     `BatchEntryKey` tuple.` Must be equality comparable and
//     `absl::Hash`-compatible. Possible types include the same member types
//     listed for `ReadRequest`. In general, the `ReadGenerationConditions` and
//     `kvstore::Key` should be specified as exactly one of a
//     `BatchEntryKeyMember` or a request member, but in some cases the key
//     might be decomposed such that some portion is included as a
//     `BatchEntryKeyMember` and the remainder is included as a request member.
template <typename DerivedDriver, typename RequestType,
          typename... BatchEntryKeyMember>
class BatchReadEntry : public Batch::Impl::Entry {
 public:
  using Driver = DerivedDriver;
  using BatchEntryKey =
      std::tuple<internal::IntrusivePtr<DerivedDriver>, BatchEntryKeyMember...>;
  using Request = RequestType;
  using KeyParam =
      std::tuple<DerivedDriver*, KeyParamType<BatchEntryKeyMember>...>;

  BatchEntryKey batch_entry_key;

  // Protected by `mutex_` until the batch is submitted.
  RequestBatch<Request> request_batch;

  DerivedDriver& driver() const { return *std::get<0>(batch_entry_key); }

  KeyParam key() const {
    return {&driver(), std::get<BatchEntryKeyMember>(batch_entry_key)...};
  }

  explicit BatchReadEntry(BatchEntryKey&& batch_entry_key)
      : Batch::Impl::Entry(std::get<0>(batch_entry_key)->BatchNestingDepth()),
        batch_entry_key(std::move(batch_entry_key)) {}

  // Adds an individual request to either an existing `BatchReadEntry` for
  // `batch`, or a new one if one does not already exist for the specified batch
  // entry key and `batch` is specified.
  //
  // If no `batch` is specified, simply creates a new `DerivedEntry` and calls
  // `Submit` immediately with `nullptr` as the batch.`
  template <typename DerivedEntry>
  static void MakeRequest(
      DerivedDriver& driver,
      KeyConstructType<BatchEntryKeyMember>... batch_entry_key_member,
      Batch::View batch, absl::Time staleness_bound, Request&& request) {
    using Self =
        BatchReadEntry<DerivedDriver, RequestType, BatchEntryKeyMember...>;
    static_assert(std::is_base_of_v<Self, DerivedEntry>);
    const auto make_entry = [&] {
      return std::make_unique<DerivedEntry>(
          BatchEntryKey(internal::IntrusivePtr<DerivedDriver>(&driver),
                        static_cast<KeyConstructType<BatchEntryKeyMember>>(
                            batch_entry_key_member)...));
    };
    if (batch) {
      Batch::Impl::From(batch)
          ->GetEntry<DerivedEntry>({&driver, batch_entry_key_member...},
                                   make_entry)
          .AddRequest(staleness_bound, std::move(request));
    } else {
      auto entry = make_entry();
      entry->request_batch.AddRequest(staleness_bound, std::move(request));
      static_cast<
          BatchReadEntry<DerivedDriver, RequestType, BatchEntryKeyMember...>*>(
          entry.release())
          ->Submit({});
    }
  }

  // Creates a new `DerivedEntry` with the same batch entry key and the
  // specified requests moved over.
  //
  // This may be useful for implementing retry logic.
  template <typename DerivedEntry>
  std::unique_ptr<DerivedEntry> MigrateExistingRequestsToNewEntry(
      span<Request> existing_requests) {
    using Self =
        BatchReadEntry<DerivedDriver, RequestType, BatchEntryKeyMember...>;
    static_assert(std::is_base_of_v<Self, DerivedEntry>);
    auto new_entry =
        std::make_unique<DerivedEntry>(BatchEntryKey(batch_entry_key));
    new_entry->request_batch.staleness_bound =
        this->request_batch.staleness_bound;
    new_entry->request_batch.requests.assign(
        std::make_move_iterator(existing_requests.begin()),
        std::make_move_iterator(existing_requests.end()));
    return new_entry;
  }

 private:
  absl::Mutex mutex_;

  void AddRequest(absl::Time staleness_bound, Request&& request) {
    absl::MutexLock lock(&mutex_);
    request_batch.AddRequest(staleness_bound, std::move(request));
  }
};

// Sets the same result for all requests in a batch.
//
// This may be useful for setting error results.
template <typename Request>
void SetCommonResult(span<const Request> requests,
                     Result<kvstore::ReadResult>&& result) {
  if (requests.empty()) return;
  for (size_t i = 1; i < requests.size(); ++i) {
    std::get<ByteRangeReadRequest>(requests[i]).promise.SetResult(result);
  }
  std::get<ByteRangeReadRequest>(requests[0])
      .promise.SetResult(std::move(result));
}
template <typename Requests>
void SetCommonResult(const Requests& requests,
                     Result<kvstore::ReadResult>&& result) {
  SetCommonResult(span<const typename Requests::value_type>(requests),
                  std::move(result));
}

template <typename Request>
void SortRequestsByStartByte(span<Request> requests) {
  std::sort(
      requests.begin(), requests.end(), [](const Request& a, const Request& b) {
        return std::get<ByteRangeReadRequest>(a).byte_range.inclusive_min <
               std::get<ByteRangeReadRequest>(b).byte_range.inclusive_min;
      });
}

// Resolves coalescsed requests with the appropriate cord subranges.
template <typename Request>
void ResolveCoalescedRequests(ByteRange coalesced_byte_range,
                              span<Request> coalesced_requests,
                              kvstore::ReadResult&& read_result) {
  for (auto& request : coalesced_requests) {
    auto& byte_range_request = std::get<ByteRangeReadRequest>(request);
    kvstore::ReadResult sub_read_result;
    sub_read_result.stamp = read_result.stamp;
    sub_read_result.state = read_result.state;
    if (read_result.state == kvstore::ReadResult::kValue) {
      assert(coalesced_byte_range.size() == read_result.value.size());
      int64_t request_start = byte_range_request.byte_range.inclusive_min -
                              coalesced_byte_range.inclusive_min;
      int64_t request_size = byte_range_request.byte_range.size();
      sub_read_result.value =
          read_result.value.Subcord(request_start, request_size);
    }
    byte_range_request.promise.SetResult(std::move(sub_read_result));
  }
}

// Determines a set of coalesced requests that will satisfy all requests in
// `requests`.
//
// \param requests Requests to attempt to coalesce. All byte ranges must have
//     already been resolved and satisfy `OptionalByteRangeRequest::IsRange()`.
// \param predicate Function with signature `bool (ByteRange
//     coalesced_byte_range, int64_t next_inclusive_min)` that determines
//     whether an additional non-overlapping byte range starting at the
//     specified offset should be coalesced with an existing (possibly
//     coalesced) byte range. Overlapping byte ranges are always coalesced.
//     Commonly a `CoalescingOptions` object may be specified as the predicate.
// \param callback Callback with signature `void (ByteRange
//     coalesced_byte_range, span<Request> coalesced_requests)` to be invoked
//     for each coalesced set of requests.
template <typename Request, typename Predicate, typename Callback>
void ForEachCoalescedRequest(span<Request> requests, Predicate predicate,
                             Callback callback) {
  SortRequestsByStartByte(requests);

  size_t request_i = 0;
  while (request_i < requests.size()) {
    auto coalesced_byte_range =
        std::get<ByteRangeReadRequest>(requests[request_i])
            .byte_range.AsByteRange();
    size_t end_request_i;
    for (end_request_i = request_i + 1; end_request_i < requests.size();
         ++end_request_i) {
      auto next_byte_range =
          std::get<ByteRangeReadRequest>(requests[end_request_i])
              .byte_range.AsByteRange();
      if (next_byte_range.inclusive_min < coalesced_byte_range.exclusive_max ||
          predicate(coalesced_byte_range, next_byte_range.inclusive_min)) {
        coalesced_byte_range.exclusive_max = std::max(
            coalesced_byte_range.exclusive_max, next_byte_range.exclusive_max);
      } else {
        break;
      }
    }
    callback(coalesced_byte_range,
             requests.subspan(request_i, end_request_i - request_i));
    request_i = end_request_i;
  }
}

// Checks that the generation constraints specified in `request` are satisfied
// by `stamp`.
//
// Returns `false` and resolves the request as aborted if the constraints are
// not satisfied.
//
// Otherwise returns `true` if the request should proceed.
template <typename Request>
bool ValidateRequestGeneration(Request& request,
                               const TimestampedStorageGeneration& stamp) {
  auto& byte_range_request = std::get<ByteRangeReadRequest>(request);
  if (!byte_range_request.promise.result_needed()) return false;
  if (!std::get<kvstore::ReadGenerationConditions>(request).Matches(
          stamp.generation)) {
    byte_range_request.promise.SetResult(
        kvstore::ReadResult::Unspecified(stamp));
    return false;
  }
  return true;
}

// Validates both the byte range and generation constraints.
//
// Returns `false` and resolves the request if invalid.
//
// Returns `true` if the request should proceed. Upon return, the byte range is
// guaranteed to satisfy `OptionalByteRangeRequest::IsRange`.
template <typename Request>
bool ValidateRequestGenerationAndByteRange(
    Request& request, const TimestampedStorageGeneration& stamp, int64_t size) {
  if (!ValidateRequestGeneration(request, stamp)) {
    return false;
  }
  auto& byte_range_request = std::get<ByteRangeReadRequest>(request);
  TENSORSTORE_ASSIGN_OR_RETURN(
      byte_range_request.byte_range,
      byte_range_request.byte_range.Validate(size),
      (byte_range_request.promise.SetResult(std::move(_)), false));
  return true;
}

// Calls `ValidateRequestGenerationAndByteRange` for every request, and removes
// requests that are resolved due to being invalid.
//
// \param requests An `std::vector` or `absl::InlinedVector` of requests.
template <typename Requests>
void ValidateGenerationsAndByteRanges(Requests& requests,
                                      const TimestampedStorageGeneration& stamp,
                                      int64_t size) {
  requests.erase(std::remove_if(requests.begin(), requests.end(),
                                [&](auto& request) {
                                  return !ValidateRequestGenerationAndByteRange(
                                      request, stamp, size);
                                }),
                 requests.end());
}

// Specifies constraints on coalescing.
//
// This may be used as a preciate for `ForEachCoalescedRequest`.
struct CoalescingOptions {
  // Maximum number of additional bytes to read per request. For example, if it
  // is estimated that the cost of an additional request is equivalent to the
  // cost of reading 256 bytes, then this should be set to 255.
  int64_t max_extra_read_bytes = 0;

  // Maximum target size for coalescing. Once this size limit is reached,
  // additional non-overlapping requests won't be added. However, this limit may
  // still be exceeded if an individual request, or set of overlapping requests,
  // exceeds this size. This can be set to balance per-request overhead with
  // additional parallelism that may be obtained from a greater number of
  // requests.
  int64_t target_coalesced_size = std::numeric_limits<int64_t>::max();

  // Checks if a new byte range starting at `next_inclusive_min` should be
  // coalesced with the existing `coalesced_byte_range`, subject to the
  // specified options.
  bool operator()(ByteRange coalesced_byte_range, int64_t next_inclusive_min) {
    return next_inclusive_min - coalesced_byte_range.exclusive_max <=
               max_extra_read_bytes &&
           coalesced_byte_range.size() < target_coalesced_size;
  }
};

constexpr CoalescingOptions kDefaultRemoteStorageCoalescingOptions = {
    /*.max_extra_read_bytes=*/4095,
    /*.target_coalesced_size=*/128 * 1024 * 10248,
};

}  // namespace internal_kvstore_batch
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_BATCH_UTIL_H_
