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

#include "tensorstore/kvstore/gcs/admission_queue.h"

#include <stddef.h>

#include <memory>
#include <optional>

#include "absl/base/call_once.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"

using ::tensorstore::internal::AnyContextResourceJsonBinder;
using ::tensorstore::internal::ContextResourceCreationContext;
using ::tensorstore::internal::ContextSpecBuilder;

namespace jb = tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_storage_gcs {

AdmissionQueue::AdmissionQueue(size_t limit)
    : limit_(limit == 0 ? std::numeric_limits<size_t>::max() : limit) {
  absl::MutexLock l(&mutex_);
  internal::intrusive_linked_list::Initialize(AdmissionAccessor{}, &head_);
}

AdmissionQueue::~AdmissionQueue() {
  absl::MutexLock l(&mutex_);
  assert(head_.next_ == &head_);
}

void AdmissionQueue::Admit(AdmissionNode* node, AdmissionNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);

  {
    absl::MutexLock lock(&mutex_);
    if (in_flight_++ >= limit_) {
      node->start_fn_ = fn;
      internal::intrusive_linked_list::InsertBefore(AdmissionAccessor{}, &head_,
                                                    node);
      return;
    }
  }
  fn(node);
}

void AdmissionQueue::Finish(AdmissionNode* node) {
  AdmissionNode* next_node = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    in_flight_--;
    next_node = head_.next_;
    if (next_node == &head_) return;
    internal::intrusive_linked_list::Remove(AdmissionAccessor{}, next_node);
  }

  // Next node gets a chance to run after clearing admission queue state.
  AdmissionNode::StartFn fn = next_node->start_fn_;
  assert(fn != nullptr);
  next_node->next_ = nullptr;
  next_node->prev_ = nullptr;
  next_node->start_fn_ = nullptr;
  fn(next_node);
}

AdmissionQueueResource::AdmissionQueueResource(size_t shared_limit)
    : shared_limit_(shared_limit) {}

AnyContextResourceJsonBinder<AdmissionQueueResource::Spec>
AdmissionQueueResource::JsonBinder() {
  return [](auto is_loading, const auto& options, auto* obj, auto* j) {
    return jb::Object(jb::Member(
        "limit",
        jb::Projection<&Spec::limit>(jb::DefaultInitializedValue(
            jb::Optional(jb::Integer<size_t>(1), [] { return "shared"; })))))(
        is_loading, options, obj, j);
  };
}

Result<AdmissionQueueResource::Resource> AdmissionQueueResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  Resource value;
  value.spec = spec;
  if (spec.limit) {
    value.queue = std::make_shared<AdmissionQueue>(*spec.limit);
  } else {
    absl::call_once(shared_once_, [&] {
      TENSORSTORE_LOG("Using default AdmissionQueue with limit ",
                      shared_limit_);
      shared_queue_ = std::make_shared<AdmissionQueue>(shared_limit_);
    });
    value.queue = shared_queue_;
  }
  return value;
}

AdmissionQueueResource::Spec AdmissionQueueResource::GetSpec(
    const Resource& resource, const ContextSpecBuilder& builder) const {
  return resource.spec;
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
