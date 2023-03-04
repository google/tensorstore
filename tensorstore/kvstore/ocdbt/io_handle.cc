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

#include "tensorstore/kvstore/ocdbt/io_handle.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

ReadonlyIoHandle::~ReadonlyIoHandle() = default;

FlushPromise::FlushPromise(FlushPromise&& other) noexcept
    : prev_linked_future_(std::move(other.prev_linked_future_)),
      promise_(std::move(other.promise_)),
      future_(std::move(other.future_)) {}

FlushPromise& FlushPromise::operator=(FlushPromise&& other) noexcept {
  prev_linked_future_ = std::move(other.prev_linked_future_);
  promise_ = std::move(other.promise_);
  future_ = std::move(other.future_);
  return *this;
}

void FlushPromise::Link(Future<const void> future) {
  if (future.null()) return;
  {
    absl::MutexLock lock(&mutex_);
    if (HaveSameSharedState(future, prev_linked_future_)) return;
    if (prev_linked_future_.null()) {
      // This is the first call to `Link` with a non-null `future`.
      prev_linked_future_ = std::move(future);
      return;
    }
    if (future_.null()) {
      // This is the first call to `Link` with a distinct non-null `future`
      // after the first call to `Link` with a non-null `future`.  We must now
      // allocate a separate `Promise`/`Future` pair and link both `future` and
      // `prev_linked_future_`.
      //
      // Note: It is safe to call `LinkError` with `mutex_` held because this
      // call cannot result in any callbacks being invoked even if one of the
      // futures is already ready.
      auto p = PromiseFuturePair<void>::LinkError(
          absl::OkStatus(), future, std::move(prev_linked_future_));
      future_ = std::move(p.future);
      promise_ = std::move(p.promise);
      prev_linked_future_ = std::move(future);
      return;
    }
  }
  prev_linked_future_ = future;
  // `LinkError` must be invoked without `mutex_` held to avoid possible
  // deadlock, since it may result in callbacks being invoked.
  LinkError(promise_, std::move(future));
}

void FlushPromise::Link(FlushPromise&& other) {
  if (other.prev_linked_future_.null()) {
    // No futures have been linked to `other`; therefore, this operation is a
    // no-op.
    return;
  }
  Future<const void> future_to_link;
  {
    absl::MutexLock lock(&mutex_);
    if (prev_linked_future_.null()) {
      // No futures have been linked to `*this`.
      *this = std::move(other);
      return;
    }
    if (promise_.null()) {
      if (!other.promise_.null()) {
        // `other` has a promise but `*this` does not.  Just move over
        // `other.promise_`.
        promise_ = std::move(other.promise_);
        future_ = std::move(other.future_);
        if (!HaveSameSharedState(prev_linked_future_,
                                 other.prev_linked_future_)) {
          future_to_link = prev_linked_future_;
          prev_linked_future_ = std::move(other.prev_linked_future_);
        }
      } else {
        // Neither `*this` nor `other` have a non-null promise.
        if (!HaveSameSharedState(prev_linked_future_,
                                 other.prev_linked_future_)) {
          // Need to create promise.
          auto p = PromiseFuturePair<void>::LinkError(
              absl::OkStatus(), std::move(prev_linked_future_),
              other.prev_linked_future_);
          future_ = std::move(p.future);
          promise_ = std::move(p.promise);
          prev_linked_future_ = std::move(other.prev_linked_future_);
        }
      }
    } else {
      if (!other.promise_.null()) {
        future_to_link = other.future_;
      } else if (!HaveSameSharedState(other.prev_linked_future_,
                                      prev_linked_future_)) {
        future_to_link = other.prev_linked_future_;
      }
      prev_linked_future_ = std::move(other.prev_linked_future_);
    }
  }
  if (!future_to_link.null()) LinkError(promise_, std::move(future_to_link));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
