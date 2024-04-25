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

#include "tensorstore/batch.h"

#include <atomic>
#include <cassert>

#include "tensorstore/batch_impl.h"
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {

Batch Batch::New() {
  Batch batch;
  batch.impl_.reset(new Impl, internal::adopt_object_ref);
  return batch;
}

void Batch::Impl::InsertIntoDepthTree(Entry& entry) {
  auto [depth_node, inserted] = nesting_depths_.FindOrInsert(
      [depth = entry.nesting_depth_](Entry& e) {
        if (depth > e.nesting_depth_) return -1;
        if (depth == e.nesting_depth_) return 0;
        return 1;
      },
      [&] { return &entry; });
  if (inserted) {
    // First entry of its nesting depth.
    entry.next_at_same_depth_ = nullptr;
  } else {
    assert(&entry != depth_node);
    entry.next_at_same_depth_ = depth_node->next_at_same_depth_;
    depth_node->next_at_same_depth_ = &entry;
    assert(entry.next_at_same_depth_ != &entry);
    assert(depth_node->next_at_same_depth_ != depth_node);
  }
}

Batch::Impl::~Impl() {
  assert(entries_.empty());
  assert(nesting_depths_.empty());
}

void Batch::SubmitBatch(ImplBase* impl_base) {
  Impl* impl = static_cast<Impl*>(impl_base);

  assert(impl->reference_count_.load(std::memory_order_relaxed) <= 1);
  impl->reference_count_.store(3, std::memory_order_relaxed);

  while (auto* start_node = impl->nesting_depths_.begin().to_pointer()) {
    impl->nesting_depths_.Remove(*start_node);
    // Traverse the linked list of nodes at this nesting depth, in order to
    // remove all nodes at this nesting depth from the hash table.  The mutex
    // does not need to be locked because there can be no concurrent accesses at
    // this point.
    {
      auto* node = start_node;
      do {
        impl->entries_.erase(node);
        node = node->next_at_same_depth_;
      } while (node);
    }

    // Traverse the linked list of nodes again, in order to submit them
    // asynchronously.
    {
      auto* node = start_node;
      do {
        auto* next = node->next_at_same_depth_;
        node->Submit(Batch::Impl::ToBatch(impl));
        node = next;
      } while (node);
    }

    if (impl->reference_count_.fetch_sub(2, std::memory_order_acq_rel) != 3) {
      // An asynchronous submit operation is still in progress.  The last
      // operation that completes will continue submitting the rest of the
      // batch.
      return;
    }

    // Restore the reference to ensure the batch isn't submitted again
    // concurrently.
    impl->reference_count_.store(3, std::memory_order_relaxed);
  }

  delete impl;
}

}  // namespace tensorstore
