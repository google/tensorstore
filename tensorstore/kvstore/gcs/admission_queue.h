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

#ifndef TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_
#define TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_

#include <assert.h>
#include <stddef.h>

#include <limits>
#include <memory>
#include <optional>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/intrusive_linked_list.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_gcs {

// AdmissionQueue and AdmissionNode implement an admission queue mechanism
// for the gcs driver. The admission queue maintains a list of pending
// `AdmissionNode` operations and managed via `AdmissionQueue::Admit` and
// `AdmissionQueue::Finish`.
//
// Generally, an AdmissionNode will also be reference counted, however the
// AdmissionQueue does not manage any reference counts. Callers should can be
// handled by adding a reference (by calling, e.g. `intrusive_ptr_increment`)
// before calling AdmissionQueue::Admit, and then taking ownership of that
// reference in the Admit functor.
//
struct AdmissionNode {
  using StartFn = void (*)(void*);

  AdmissionNode* next_ = nullptr;
  AdmissionNode* prev_ = nullptr;
  StartFn start_fn_ = nullptr;
};

using AdmissionAccessor = internal::intrusive_linked_list::MemberAccessor<
    AdmissionNode, &AdmissionNode::prev_, &AdmissionNode::next_>;

class AdmissionQueue {
 public:
  // Create an admission queue with the given limit.
  AdmissionQueue(size_t limit);
  ~AdmissionQueue();

  // Admit a task node to the queue.  If the node is admitted, then the
  // start function, `fn(node)`, is invoked, otherwise the admission queue will
  // arrange to invoke `fn(node)` at some later time.
  void Admit(AdmissionNode* node, AdmissionNode::StartFn fn);

  // Mark a task node for completion. When a node finishes in this way, a queued
  // node will have it's start function invoked.
  void Finish(AdmissionNode* node);

 private:
  const size_t limit_;
  size_t in_flight_ = 0;
  absl::Mutex mutex_;
  AdmissionNode head_ ABSL_GUARDED_BY(mutex_);
};

/// Specifies an admission queue as a resource, compatible with a concurrency
/// resource.
struct AdmissionQueueResource {
 public:
  AdmissionQueueResource(size_t shared_limit);

  struct Spec {
    // If equal to `nullopt`, indicates that the shared executor is used.
    std::optional<size_t> limit;
  };
  struct Resource {
    Spec spec;
    std::shared_ptr<AdmissionQueue> queue;
  };

  static Spec Default() { return Spec{std::nullopt}; }

  static internal::AnyContextResourceJsonBinder<Spec> JsonBinder();

  Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) const;

  Spec GetSpec(const Resource& resource,
               const internal::ContextSpecBuilder& builder) const;

 private:
  /// Size of thread pool referenced by `shared_queue_`.
  size_t shared_limit_;
  /// Protects initialization of `shared_queue_`.
  mutable absl::once_flag shared_once_;
  /// Lazily-initialized shared AdmissionQueue used by default spec.
  mutable std::shared_ptr<AdmissionQueue> shared_queue_;
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_
