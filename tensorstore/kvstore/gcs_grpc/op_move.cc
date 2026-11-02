// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs_grpc/op_move.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"

// proto
#include "google/storage/v2/storage.pb.h"

using ::google::storage::v2::GetObjectRequest;
using ::google::storage::v2::MoveObjectRequest;
using ::tensorstore::internal::GrpcStatusToAbslStatus;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT internal_metrics::Counter<int64_t> move_fallback_metric;

TENSORSTORE_GLOBAL_INITIALIZER {
  auto& r = internal_metrics::GetMetricRegistry();
  r.Register(&move_fallback_metric,
             internal_metrics::MetricMetadata(
                 "/tensorstore/kvstore/gcs_grpc/move_fallback",
                 "gcs_grpc Count of retried MoveObject requests falling back "
                 "to GetObject"));
}

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

// Implements Storage::MoveObject RPC.
struct MoveTask : public internal::AtomicReferenceCount<MoveTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  kvstore::WriteOptions options_;
  StorageGeneration source_generation_;
  // Identifies the source object; sent as the idempotency token and compared
  // against the destination's `kWriteIdMetadataKey` when the outcome of a
  // retried move is ambiguous.
  std::string write_id_;
  Promise<TimestampedStorageGeneration> promise_;

  // Working state
  const absl::Time start_time_ = absl::Now();
  MoveObjectRequest request_;
  GetObjectRequest get_request_;
  ::google::storage::v2::Object response_;
  // Whether an earlier `MoveObject` RPC was already put on the wire and is
  // being retried.
  bool retrying_ = false;
  // Whether this task has transitioned to querying the destination object via
  // `GetObject` after a retried `MoveObject` returned `NOT_FOUND`.
  bool querying_destination_ = false;
  GrpcCallContext call_;

  MoveTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
           kvstore::WriteOptions options, StorageGeneration source_generation,
           std::string write_id, Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)),
        options_(std::move(options)),
        source_generation_(std::move(source_generation)),
        write_id_(std::move(write_id)),
        promise_(std::move(promise)),
        call_(write_id_) {
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<MoveTask>(
                                       this)] { self->call_.TryCancel(); });
  }

  void SetupRequest(std::string_view bucket_name,
                    std::string_view source_object,
                    std::string_view destination_object) {
    request_.set_bucket(bucket_name);
    request_.set_source_object(source_object);
    request_.set_destination_object(destination_object);
    if (auto gen = EncodeGenerationCondition(source_generation_)) {
      request_.set_if_source_generation_match(*gen);
    }
    if (auto gen = EncodeGenerationCondition(
            options_.generation_conditions.if_equal)) {
      request_.set_if_generation_match(*gen);
    }
  }

  void Start();
  void RunAttempt();
  void RunWithContext(std::shared_ptr<grpc::ClientContext> context);
  void AttemptFinished(absl::Status status);
  void FinishMove(absl::Status status, bool was_retrying);
  void FinishQueryDestination(absl::Status status);
};

void MoveTask::Start() {
  ABSL_LOG_IF(INFO, gcs_grpc_logging)
      << "MoveTask " << request_.source_object() << " -> "
      << request_.destination_object();
  RunAttempt();
}

void MoveTask::RunAttempt() {
  if (!promise_.result_needed()) {
    return;
  }

  GrpcCallContext::RunWithContext(internal::IntrusivePtr<MoveTask>(this),
                                  driver_.get(), /*routing_token=*/{},
                                  &MoveTask::RunWithContext,
                                  &MoveTask::AttemptFinished);
}

void MoveTask::RunWithContext(std::shared_ptr<grpc::ClientContext> context) {
  if (!promise_.result_needed() || !call_.BeginAttempt(context)) {
    return;
  }

  response_.Clear();
  auto stub = driver_->get_stub();

  intrusive_ptr_increment(this);  // Adopted by callback
  auto callback = WithExecutor(driver_->executor(), [this](grpc::Status s) {
    internal::IntrusivePtr<MoveTask> self(this, internal::adopt_object_ref);
    self->AttemptFinished(GrpcStatusToAbslStatus(s));
  });

  if (!querying_destination_) {
    ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2))
        << this << " " << ConciseDebugString(request_);
    stub->async()->MoveObject(context.get(), &request_, &response_,
                              std::move(callback));
  } else {
    ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2))
        << this << " " << ConciseDebugString(get_request_);
    stub->async()->GetObject(context.get(), &get_request_, &response_,
                             std::move(callback));
  }
}

void MoveTask::AttemptFinished(absl::Status status) {
  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2)) << this << " " << status;

  call_.EndAttempt();
  if (!promise_.result_needed()) {
    return;
  }

  const bool was_retrying = retrying_;
  if (!querying_destination_ && !status.ok() && IsRetriable(status)) {
    retrying_ = true;
  }
  if (call_.MaybeRetry(driver_.get(), status,
                       [self = internal::IntrusivePtr<MoveTask>(this)] {
                         self->RunAttempt();
                       })) {
    return;
  }

  if (!querying_destination_) {
    FinishMove(std::move(status), was_retrying);
  } else {
    FinishQueryDestination(std::move(status));
  }
}

void MoveTask::FinishMove(absl::Status status, bool was_retrying) {
  // If `MoveObject` was retried and failed with `NOT_FOUND` (for example,
  // because the backend did not honor `x-goog-gcs-idempotency-token` or the
  // idempotency window expired), the source temporary object may have been
  // consumed by an earlier attempt whose response was lost. Query the
  // destination object metadata: the move is attributed to this task only if
  // the destination carries this task's write id, since a concurrent writer
  // may have replaced the destination in the meantime.
  if (was_retrying && absl::IsNotFound(status)) {
    move_fallback_metric.Increment();
    querying_destination_ = true;
    get_request_.set_bucket(request_.bucket());
    get_request_.set_object(request_.destination_object());
    call_.ResetAttempts();
    RunAttempt();
    return;
  }

  TimestampedStorageGeneration r;
  r.time = start_time_;
  if (IsWritePreconditionMismatch(status,
                                  options_.generation_conditions.if_equal)) {
    r.generation = StorageGeneration::Unknown();
    promise_.SetResult(std::move(r));
    return;
  }
  if (!status.ok()) {
    promise_.SetResult(std::move(status));
    return;
  }
  if (response_.generation() == 0) {
    promise_.SetResult(absl::InternalError(absl::StrFormat(
        "MoveObject completed with zero generation for %v",
        tensorstore::QuoteString(request_.destination_object()))));
    return;
  }
  r.generation = StorageGeneration::FromUint64(response_.generation());
  promise_.SetResult(std::move(r));
}

void MoveTask::FinishQueryDestination(absl::Status status) {
  TimestampedStorageGeneration r;
  r.time = start_time_;
  if (absl::IsNotFound(status) &&
      !StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
    r.generation = StorageGeneration::Unknown();
    promise_.SetResult(std::move(r));
    return;
  }
  if (!status.ok()) {
    promise_.SetResult(std::move(status));
    return;
  }
  if (response_.generation() == 0) {
    promise_.SetResult(absl::InternalError(absl::StrFormat(
        "GetObject completed with zero generation for %v",
        tensorstore::QuoteString(request_.destination_object()))));
    return;
  }
  auto it = response_.metadata().find(std::string(kWriteIdMetadataKey));
  if (it == response_.metadata().end() || it->second != write_id_) {
    // The destination was not produced by this move: either the earlier
    // attempt never applied, or a concurrent writer replaced the destination
    // afterwards.  Reporting another writer's generation as ours would let
    // the caller cache our value under their generation.
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      r.generation = StorageGeneration::Unknown();
      promise_.SetResult(std::move(r));
      return;
    }
    promise_.SetResult(absl::AbortedError(absl::StrFormat(
        "MoveObject outcome for %v is ambiguous after retry",
        tensorstore::QuoteString(request_.destination_object()))));
    return;
  }
  r.generation = StorageGeneration::FromUint64(response_.generation());
  promise_.SetResult(std::move(r));
}

}  // namespace

Future<TimestampedStorageGeneration> InitiateMoveObject(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    std::string source_object, std::string destination_object,
    kvstore::WriteOptions options, StorageGeneration source_generation,
    std::string write_id) {
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();
  auto task = internal::MakeIntrusivePtr<MoveTask>(
      std::move(driver), std::move(options), std::move(source_generation),
      std::move(write_id), std::move(op.promise));
  task->SetupRequest(task->driver_->bucket_name(), source_object,
                     destination_object);
  task->Start();
  return std::move(op.future);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
