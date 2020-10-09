// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_
#define TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_

/// \file
///
/// Framework for defining writeback caches that aggregate multiple independent
/// atomic write operations.

#include <vector>

#include "tensorstore/internal/async_cache.h"
#include "tensorstore/internal/cache.h"
#include "tensorstore/internal/estimate_heap_usage.h"

namespace tensorstore {
namespace internal {

/// CRTP base class for defining an `AsyncCache` that aggregates
/// multiple independent atomic write operations.
///
/// This class does not itself implement any specific writeback behavior, but
/// keeps track of the lists of pending and issued `PendingWrite` operations.
///
/// \tparam Derived The derived class type, must define a `PendingWrite` type
///     and optionally override `DoGetPendingWriteSize`.
/// \tparam Parent The base class to inherit from, must inherit from (or equal)
///     `AsyncCache`.
template <typename Derived, typename Parent>
class AggregateWritebackCache : public Parent {
 public:
  class TransactionNode : public Parent::TransactionNode {
   public:
    using Parent::TransactionNode::TransactionNode;

    using PendingWrite = typename Derived::PendingWrite;
    std::vector<PendingWrite> pending_writes;

    /// Adds a `PendingWrite` to the `pending_writes` list, then calls
    /// `FinishWrite`.
    ///
    /// \param pending_write The operation to add.
    /// \param write_flags Flags to pass to `FinishWrite`.
    void AddPendingWrite(PendingWrite pending_write) {
      pending_writes_size +=
          this->ComputePendingWriteSizeInBytes(pending_write);
      pending_writes.emplace_back(std::move(pending_write));
      this->MarkSizeUpdated();
    }

    /// Returns the additional heap memory required by a `PendingWrite` object
    /// (should not include `sizeof(PendingWrite)`).
    ///
    /// The `Derived::TransactionNode` class should override this method if
    /// additional heap memory is required.
    virtual size_t ComputePendingWriteSizeInBytes(
        const PendingWrite& pending_write) {
      return 0;
    }

    /// Computes the size from the cached `pending_writes_size` value.
    size_t ComputeWriteStateSizeInBytes() override {
      return internal::EstimateHeapUsage(pending_writes,
                                         /*include_children=*/false) +
             pending_writes_size;
    }

    void WritebackError() override {
      pending_writes.clear();
      this->Parent::TransactionNode::WritebackError();
    }

   private:
    friend class AggregateWritebackCache;
    /// Additional heap-allocated memory required by `pending_writes`, not
    /// including the memory allocated directly by the `std::vector`.
    size_t pending_writes_size = 0;
  };

  using Parent::Parent;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_AGGREGATE_WRITEBACK_CACHE_H_
