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

#include "tensorstore/kvstore/gcs_grpc/op_delete.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

// proto
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.pb.h"

using ::tensorstore::internal::GrpcStatusToAbslStatus;

using ::google::storage::v2::DeleteObjectRequest;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

// Implements GcsGrpcKeyValueStore::Delete
// rpc DeleteObject(DeleteObjectRequest) returns (google.protobuf.Empty) {}
struct DeleteTask : public internal::AtomicReferenceCount<DeleteTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::WriteOptions options_;
  Promise<TimestampedStorageGeneration> promise_;

  // Working state
  absl::Time start_time_;
  DeleteObjectRequest request_;
  ::google::protobuf::Empty response_;
  int attempt_ = 0;
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  DeleteTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
             kvstore::WriteOptions options,
             Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)),
        options_(std::move(options)),
        promise_(std::move(promise)) {}

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  void Start(std::string_view bucket_name, std::string_view object_name)
      ABSL_LOCKS_EXCLUDED(mutex_);
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_);
  void RetryWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_);
  void DeleteFinished(absl::Status status) ABSL_LOCKS_EXCLUDED(mutex_);
};

void DeleteTask::Start(std::string_view bucket_name,
                       std::string_view object_name) {
  ABSL_LOG_IF(INFO, gcs_grpc_logging) << "DeleteTask " << object_name;

  promise_.ExecuteWhenNotNeeded(
      [self = internal::IntrusivePtr<DeleteTask>(this)] { self->TryCancel(); });

  request_.set_bucket(bucket_name);
  request_.set_object(object_name);
  if (!StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
    auto gen =
        StorageGeneration::ToUint64(options_.generation_conditions.if_equal);
    request_.set_if_generation_match(gen);
  }
  Retry();
}

void DeleteTask::Retry() {
  if (!promise_.result_needed()) {
    return;
  }

  auto context_future = driver_->AllocateContext();
  context_future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<DeleteTask>(this),
       context_future](ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
        self->RetryWithContext(std::move(f).value());
      });
  context_future.Force();
}

void DeleteTask::RetryWithContext(
    std::shared_ptr<grpc::ClientContext> context) {
  start_time_ = absl::Now();
  {
    absl::MutexLock lock(mutex_);
    assert(context_ == nullptr);
    context_ = std::move(context);
    auto stub = driver_->get_stub();

    intrusive_ptr_increment(this);  // Adopted by OnDone
    stub->async()->DeleteObject(
        context_.get(), &request_, &response_,
        WithExecutor(driver_->executor(), [this](::grpc::Status s) {
          internal::IntrusivePtr<DeleteTask> self(this,
                                                  internal::adopt_object_ref);
          self->DeleteFinished(GrpcStatusToAbslStatus(s));
        }));
  }
}

void DeleteTask::DeleteFinished(absl::Status status) {
  if (!promise_.result_needed()) {
    return;
  }

  {
    absl::MutexLock lock(mutex_);
    context_ = nullptr;
  }

  if (!status.ok() && attempt_ == 0 &&
      status.code() == absl::StatusCode::kUnauthenticated) {
    // Allow a single unauthenticated error.
    attempt_++;
    Retry();
    return;
  }
  if (!status.ok() && IsRetriable(status)) {
    status = driver_->BackoffForAttemptAsync(
        std::move(status), attempt_++,
        [self = internal::IntrusivePtr<DeleteTask>(this)] { self->Retry(); });
    if (status.ok()) {
      return;
    }
  }

  TimestampedStorageGeneration r;
  r.time = start_time_;
  r.generation = StorageGeneration::NoValue();
  if (absl::IsFailedPrecondition(status)) {
    // precondition did not match.
    r.generation = StorageGeneration::Unknown();
  } else if (absl::IsNotFound(status)) {
    // object missing; that's probably ok.
    if (!options_.generation_conditions.MatchesNoValue()) {
      r.generation = StorageGeneration::Unknown();
    }
  } else if (!status.ok()) {
    promise_.SetResult(std::move(status));
    return;
  }
  promise_.SetResult(std::move(r));
}

}  // namespace

/// Implements GcsGrpcKeyValueStore::Delete
Future<TimestampedStorageGeneration> InitiateDelete(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver, kvstore::Key key,
    kvstore::WriteOptions options) {
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  auto task = internal::MakeIntrusivePtr<DeleteTask>(
      std::move(driver), std::move(options), std::move(op.promise));
  task->Start(task->driver_->bucket_name(), key);
  return std::move(op.future);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
