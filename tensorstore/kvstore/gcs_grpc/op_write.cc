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

#include "tensorstore/kvstore/gcs_grpc/op_write.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

// proto
#include "google/storage/v2/storage.pb.h"

using ::tensorstore::internal::GrpcStatusToAbslStatus;

using ::google::storage::v2::WriteObjectRequest;
using ::google::storage::v2::WriteObjectResponse;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

static constexpr size_t kMaxWriteBytes =
    google::storage::v2::ServiceConstants::MAX_WRITE_CHUNK_BYTES;

absl::AnyInvocable<void(grpc::ClientContext*,
                        google::storage::v2::WriteObjectRequest&)>
    g_test_hook;

// Implements GcsGrpcKeyValueStore::Write
// rpc WriteObject(stream WriteObjectRequest) returns (WriteObjectResponse) {}
// rpc StartResumableWrite(StartResumableWriteRequest) returns
// (StartResumableWriteResponse) {}
struct WriteTask : public internal::AtomicReferenceCount<WriteTask>,
                   public grpc::ClientWriteReactor<WriteObjectRequest> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  internal_kvstore::CommonMetrics& common_metrics_;
  std::string bucket_name_;
  std::string object_name_;

  kvstore::WriteOptions options_;
  absl::Cord value_;
  size_t value_offset_;
  absl::Time start_time_;
  absl::crc32c_t crc32c_;
  Promise<TimestampedStorageGeneration> promise_;

  // working state.
  WriteObjectRequest request_;
  WriteObjectResponse response_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  WriteTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
            internal_kvstore::CommonMetrics& common_metrics,
            std::string object_name, kvstore::WriteOptions options,
            absl::Cord data, Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)),
        common_metrics_(common_metrics),
        bucket_name_(driver_->bucket_name()),
        object_name_(std::move(object_name)),
        options_(std::move(options)),
        value_(std::move(data)),
        value_offset_(0),
        crc32c_(absl::crc32c_t{0}),
        promise_(std::move(promise)) {}

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  Result<TimestampedStorageGeneration> HandleFinalStatus(
      absl::Status status, WriteObjectResponse& response);
  void UpdateRequestForNextWrite(WriteObjectRequest& request);

  void OnWriteDone(bool ok) override;
  void OnDone(const grpc::Status& s) override;

  void Start() ABSL_LOCKS_EXCLUDED(mutex_);
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_);
  void RetryWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_);
  void WriteFinished(absl::Status status) ABSL_LOCKS_EXCLUDED(mutex_);
};

Result<TimestampedStorageGeneration> WriteTask::HandleFinalStatus(
    absl::Status status, WriteObjectResponse& response) {
  TimestampedStorageGeneration result;
  result.time = start_time_;
  if (response.has_resource()) {
    result.generation =
        StorageGeneration::FromUint64(response.resource().generation());
  }
  if (absl::IsFailedPrecondition(status) || absl::IsAlreadyExists(status)) {
    // if_equal condition did not match.
    result.generation = StorageGeneration::Unknown();
  } else if (absl::IsNotFound(status) &&
             !StorageGeneration::IsUnknown(
                 options_.generation_conditions.if_equal)) {
    // precondition did not match.
    result.generation = StorageGeneration::Unknown();
  } else if (!status.ok()) {
    return status;
  }
  return std::move(result);
}

void WriteTask::UpdateRequestForNextWrite(WriteObjectRequest& request) {
  if (value_offset_ == 0) {
    request.mutable_write_object_spec()->set_object_size(value_.size());
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      auto gen =
          StorageGeneration::ToUint64(options_.generation_conditions.if_equal);
      request.mutable_write_object_spec()->set_if_generation_match(gen);
    }
  } else {
    // After the first request, clear the spec.
    request.clear_write_object_spec();
  }

  request.set_write_offset(value_offset_);
  auto next_part = value_.Subcord(value_offset_, kMaxWriteBytes);
  auto& checksummed_data = *request.mutable_checksummed_data();
  checksummed_data.set_content(next_part);
  auto chunk_crc32c = ComputeCrc32c(checksummed_data.content());
  checksummed_data.set_crc32c(static_cast<uint32_t>(chunk_crc32c));
  crc32c_ = absl::ConcatCrc32c(crc32c_, chunk_crc32c,
                               checksummed_data.content().size());
  value_offset_ = value_offset_ + next_part.size();
  if (value_offset_ == value_.size()) {
    /// This is the last request.
    request.mutable_object_checksums()->set_crc32c(
        static_cast<uint32_t>(crc32c_));
    request.set_finish_write(true);
  }
}

void WriteTask::Start() {
  ABSL_LOG_IF(INFO, gcs_grpc_logging) << "WriteTask " << object_name_;
  promise_.ExecuteWhenNotNeeded(
      [self = internal::IntrusivePtr<WriteTask>(this)] { self->TryCancel(); });
  Retry();
}

// Retry/Continue a call.
void WriteTask::Retry() {
  if (!promise_.result_needed()) {
    return;
  }

  // Retry always starts from a "clean" request, so clear the value.
  request_.Clear();
  value_offset_ = 0;
  crc32c_ = absl::crc32c_t{0};
  start_time_ = absl::Now();

  // First request, make sure that the spec is setup correctly.
  auto& resource = *request_.mutable_write_object_spec()->mutable_resource();
  resource.set_bucket(bucket_name_);
  resource.set_name(object_name_);

  auto context_future = driver_->AllocateContext();
  context_future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<WriteTask>(this),
       context_future](ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
        self->RetryWithContext(std::move(f).value());
      });
  context_future.Force();
}

void WriteTask::RetryWithContext(std::shared_ptr<grpc::ClientContext> context) {
  if (g_test_hook) {
    g_test_hook(context.get(), request_);
  }

  {
    absl::MutexLock lock(mutex_);
    assert(context_ == nullptr);
    context_ = std::move(context);
    auto stub = driver_->get_stub();
    // Initiate the write.
    intrusive_ptr_increment(this);
    stub->async()->WriteObject(context_.get(), &response_, this);
  }

  UpdateRequestForNextWrite(request_);

  auto options = grpc::WriteOptions();
  if (request_.finish_write()) {
    options.set_last_message();
  }
  StartWrite(&request_, options);
  StartCall();
}

void WriteTask::OnWriteDone(bool ok) {
  // Not streaming any additional data bits.
  if (!ok) return;
  if (request_.finish_write()) return;

  UpdateRequestForNextWrite(request_);

  auto options = grpc::WriteOptions();
  if (request_.finish_write()) {
    options.set_last_message();
  }
  StartWrite(&request_, options);
}

void WriteTask::OnDone(const grpc::Status& s) {
  internal::IntrusivePtr<WriteTask> self(this, internal::adopt_object_ref);
  driver_->executor()(
      [self = std::move(self), status = GrpcStatusToAbslStatus(s)] {
        self->WriteFinished(std::move(status));
      });
}

void WriteTask::WriteFinished(absl::Status status) {
  if (!promise_.result_needed()) {
    return;
  }

  auto latency = absl::Now() - start_time_;
  common_metrics_.write_latency_ms.Observe(absl::ToInt64Milliseconds(latency));
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
        [self = internal::IntrusivePtr<WriteTask>(this)] { self->Retry(); });
    if (status.ok()) {
      return;
    }
  }

  promise_.SetResult(HandleFinalStatus(status, response_));
}

}  // namespace

void SetTestHook(
    absl::AnyInvocable<void(grpc::ClientContext*,
                            google::storage::v2::WriteObjectRequest&)>&&
        debug_hook) {
  g_test_hook = std::move(debug_hook);
}

Future<TimestampedStorageGeneration> InitiateWrite(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    internal_kvstore::CommonMetrics& common_metrics, kvstore::Key key,
    kvstore::Value data, kvstore::WriteOptions options) {
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  auto task = internal::MakeIntrusivePtr<WriteTask>(
      std::move(driver), common_metrics, std::move(key), std::move(options),
      std::move(data), std::move(op.promise));
  task->Start();
  return std::move(op.future);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
