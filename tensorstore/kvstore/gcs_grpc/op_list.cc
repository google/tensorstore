// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs_grpc/op_list.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"

// proto
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"

using ::tensorstore::internal::GrpcStatusToAbslStatus;

using ::google::storage::v2::ListObjectsRequest;
using ::google::storage::v2::ListObjectsResponse;
using ::google::storage::v2::Storage;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

// Implements GcsGrpcKeyValueStore::List
// rpc ListObjects(ListObjectsRequest) returns (ListObjectsResponse) {}
struct ListTask : public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::ListOptions options_;
  kvstore::ListReceiver receiver_;

  // working state.
  std::shared_ptr<Storage::StubInterface> stub_;
  ListObjectsRequest request;
  ListObjectsResponse response;
  GrpcCallContext call_;

  ListTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
           kvstore::ListOptions options, kvstore::ListReceiver receiver)
      : driver_(std::move(driver)),
        options_(std::move(options)),
        receiver_(std::move(receiver)) {
    execution::set_starting(receiver_, [this] { call_.TryCancel(); });
  }

  ~ListTask() {
    call_.EndAttempt();
    driver_ = {};
    execution::set_stopping(receiver_);
  }

  void Start(std::string_view bucket_name);
  void Retry();
  void RetryWithContext(std::shared_ptr<grpc::ClientContext> context);
  void ListFinished(absl::Status status);
};

void ListTask::Start(std::string_view bucket_name) {
  ABSL_LOG_IF(INFO, gcs_grpc_logging) << "ListTask " << options_.range;

  request.set_lexicographic_start(options_.range.inclusive_min);
  request.set_lexicographic_end(options_.range.exclusive_max);
  request.set_parent(bucket_name);
  request.set_page_size(1000);  // maximum.

  Retry();
}

// Retry/Continue a call.
void ListTask::Retry() {
  if (call_.is_cancelled()) {
    execution::set_done(receiver_);
    return;
  }
  GrpcCallContext::RunWithContext(internal::IntrusivePtr<ListTask>(this),
                                  driver_.get(), /*routing_token=*/{},
                                  &ListTask::RetryWithContext,
                                  &ListTask::ListFinished);
}

void ListTask::RetryWithContext(std::shared_ptr<grpc::ClientContext> context) {
  if (!call_.BeginAttempt(context)) {
    execution::set_done(receiver_);
    return;
  }
  stub_ = driver_->get_stub();

  intrusive_ptr_increment(this);
  stub_->async()->ListObjects(
      context.get(), &request, &response,
      WithExecutor(driver_->executor(), [this](::grpc::Status s) {
        internal::IntrusivePtr<ListTask> self(this, internal::adopt_object_ref);
        self->ListFinished(GrpcStatusToAbslStatus(s));
      }));
}

void ListTask::ListFinished(absl::Status status) {
  call_.EndAttempt();

  if (call_.is_cancelled()) {
    execution::set_done(receiver_);
    return;
  }

  if (call_.MaybeRetry(
          driver_.get(), status,
          [self = internal::IntrusivePtr<ListTask>(this)] { self->Retry(); })) {
    return;
  }

  if (!status.ok()) {
    execution::set_error(receiver_, std::move(status));
    return;
  }

  bool done = false;
  for (const auto& o : response.objects()) {
    if (call_.is_cancelled()) {
      done = true;
      break;
    }
    std::string_view name = o.name();
    if (!Contains(options_.range, name)) {
      if (KeyRange::CompareKeyAndExclusiveMax(
              name, options_.range.exclusive_max) >= 0) {
        done = true;
        break;
      }
      continue;
    }
    if (options_.strip_prefix_length) {
      if (options_.strip_prefix_length > name.size()) {
        continue;
      }
      name = name.substr(options_.strip_prefix_length);
    }
    execution::set_value(receiver_,
                         kvstore::ListEntry{
                             std::string(name),
                             kvstore::ListEntry::checked_size(o.size()),
                         });
  }
  if (!done && !response.next_page_token().empty()) {
    // If there is a continuation token, issue the next request.
    request.set_page_token(response.next_page_token());
    response.Clear();
    call_.ResetAttempts();
    Retry();
    return;
  }

  execution::set_done(receiver_);
}

}  // namespace

void InitiateList(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
                  kvstore::ListOptions options,
                  kvstore::ListReceiver receiver) {
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }

  auto task = internal::MakeIntrusivePtr<ListTask>(
      std::move(driver), std::move(options), std::move(receiver));
  task->Start(task->driver_->bucket_name());
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
