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

#include "tensorstore/kvstore/gcs_grpc/op_read.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
#include "riegeli/bytes/cord_writer.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

// proto
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.pb.h"

using ::tensorstore::internal::GrpcStatusToAbslStatus;

using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::ReadObjectResponse;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

absl::AnyInvocable<void(grpc::ClientContext*,
                        google::storage::v2::ReadObjectRequest&)>
    g_test_hook;

// Implements GcsGrpcKeyValueStore::Read
// rpc ReadObject(ReadObjectRequest) returns (stream ReadObjectResponse) {}
struct ReadTask : public internal::AtomicReferenceCount<ReadTask>,
                  public grpc::ClientReadReactor<ReadObjectResponse> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  internal_kvstore::CommonMetrics& common_metrics_;
  std::string bucket_name_;
  Promise<kvstore::ReadResult> promise_;

  // Read options
  kvstore::ReadOptions options_;

  // Working state.
  TimestampedStorageGeneration storage_generation_;
  std::optional<absl::crc32c_t> crc32c_;
  absl::crc32c_t combined_crc32c_ = absl::crc32c_t(0);
  absl::Cord value_;
  riegeli::CordWriter<> cord_writer_{&value_};

  ReadObjectRequest request_;
  ReadObjectResponse response_;

  int attempt_ = 0;
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);

  ReadTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
           internal_kvstore::CommonMetrics& common_metrics,
           kvstore::ReadOptions options, Promise<kvstore::ReadResult> promise)
      : driver_(std::move(driver)),
        common_metrics_(common_metrics),
        promise_(std::move(promise)),
        options_(std::move(options)) {
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->TryCancel(); });
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  void SetupRequest(std::string_view bucket_name, std::string_view object_name);
  absl::Status HandleResponse(ReadObjectResponse& response);
  Result<kvstore::ReadResult> HandleFinalStatus(absl::Status status);

  void Start() ABSL_LOCKS_EXCLUDED(mutex_);
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_);
  void RetryWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_);
  void OnReadDone(bool ok) override;
  void OnDone(const grpc::Status& s) override;
  void ReadFinished(absl::Status status);
};

void ReadTask::SetupRequest(std::string_view bucket_name,
                            std::string_view object_name) {
  request_.set_bucket(bucket_name);
  request_.set_object(object_name);

  if (!StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
    uint64_t gen =
        StorageGeneration::IsNoValue(options_.generation_conditions.if_equal)
            ? 0
            : StorageGeneration::ToUint64(
                  options_.generation_conditions.if_equal);
    request_.set_if_generation_match(gen);
  }
  if (!StorageGeneration::IsUnknown(
          options_.generation_conditions.if_not_equal)) {
    uint64_t gen = StorageGeneration::IsNoValue(
                       options_.generation_conditions.if_not_equal)
                       ? 0
                       : StorageGeneration::ToUint64(
                             options_.generation_conditions.if_not_equal);
    request_.set_if_generation_not_match(gen);
  }
  if (options_.byte_range.inclusive_min != 0) {
    request_.set_read_offset(options_.byte_range.inclusive_min);
  }
  if (options_.byte_range.exclusive_max != -1) {
    auto target_size = options_.byte_range.size();
    assert(target_size >= 0);
    // read_limit == 0 reads the entire object; for a 0-byte read
    // request 1 byte and we'll return an empty cord in HandleFinalStatus.
    request_.set_read_limit(target_size == 0 ? 1 : target_size);
  }
}

absl::Status ReadTask::HandleResponse(ReadObjectResponse& response) {
  if (response.has_metadata()) {
    storage_generation_.generation =
        StorageGeneration::FromUint64(response.metadata().generation());
  }
  if (response.has_object_checksums() &&
      response.object_checksums().crc32c() != 0 &&
      options_.byte_range.inclusive_min == 0 &&
      !options_.byte_range.exclusive_max) {
    // Do not validate byte-range requests.
    crc32c_ = absl::crc32c_t(response.object_checksums().crc32c());
  }
  if (response.has_content_range()) {
    // The content-range request indicates the expected data. If it does not
    // satisfy the byte range request, cancel the read with an error. Allow
    // the returned size to exceed the requested size.
    auto returned_size =
        response.content_range().end() - response.content_range().start();
    if (auto size = options_.byte_range.size();
        (size > 0 && size != returned_size) ||
        (options_.byte_range.inclusive_min >= 0 &&
         response.content_range().start() !=
             options_.byte_range.inclusive_min)) {
      return absl::OutOfRangeError(
          tensorstore::StrCat("Requested byte range ", options_.byte_range,
                              " was not satisfied by GCS object with size ",
                              response.content_range().complete_length()));
    }
  }
  if (response.has_checksummed_data() &&
      !response.checksummed_data().content().empty()) {
    const auto& content = response.checksummed_data().content();
    // Validate the content checksum.
    if (response.checksummed_data().has_crc32c()) {
      absl::crc32c_t expected_crc32c =
          absl::crc32c_t(response.checksummed_data().crc32c());
      absl::crc32c_t chunk_crc32c = ComputeCrc32c(content);
      if (chunk_crc32c != expected_crc32c) {
        return absl::DataLossError(absl::StrFormat(
            "Object fragment crc32c %08x does not match expected crc32c %08x",
            static_cast<uint32_t>(chunk_crc32c),
            static_cast<uint32_t>(expected_crc32c)));
      }
      combined_crc32c_ =
          absl::ConcatCrc32c(combined_crc32c_, chunk_crc32c, content.size());
    } else {
      // This chunk missed crc32c data; clear the expected crc32c value.
      crc32c_ = std::nullopt;
    }
    if (!cord_writer_.Write(content)) {
      return cord_writer_.status();
    }
  }
  return absl::OkStatus();
}

Result<kvstore::ReadResult> ReadTask::HandleFinalStatus(absl::Status status) {
  if (absl::IsFailedPrecondition(status) || absl::IsAborted(status)) {
    // Failed precondition is set when either the if_generation_match or
    // the if_generation_not_match fails.
    if (!StorageGeneration::IsUnknown(
            options_.generation_conditions.if_equal)) {
      storage_generation_.generation = StorageGeneration::Unknown();
    } else {
      storage_generation_.generation =
          options_.generation_conditions.if_not_equal;
    }
    return kvstore::ReadResult::Unspecified(std::move(storage_generation_));
  } else if (absl::IsNotFound(status)) {
    return kvstore::ReadResult::Missing(storage_generation_.time);
  } else if (!status.ok()) {
    return status;
  }

  if (StorageGeneration::IsUnknown(storage_generation_.generation)) {
    // Bad metadata was returned by BlobService; this is unexpected, and
    // usually indicates a bug in our testing.
    return absl::InternalError("Object missing a valid generation");
  }

  // Validate the content checksum.
  if (crc32c_ && combined_crc32c_ != *crc32c_) {
    return absl::DataLossError(absl::StrFormat(
        "Object crc32c %08x does not match expected crc32c %08x",
        static_cast<uint32_t>(combined_crc32c_),
        static_cast<uint32_t>(*crc32c_)));
  }

  // Validate the content checksum.
  if (!cord_writer_.Close()) {
    return cord_writer_.status();
  }
  if (options_.byte_range.size() == 0) {
    return kvstore::ReadResult::Value({}, std::move(storage_generation_));
  }
  return kvstore::ReadResult::Value(std::move(value_),
                                    std::move(storage_generation_));
}

void ReadTask::Start() {
  ABSL_LOG_IF(INFO, gcs_grpc_logging)
      << this << " ReadTask " << request_.object();

  auto context_future = driver_->AllocateContext();
  context_future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<ReadTask>(this),
       context_future](ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
        self->RetryWithContext(std::move(f).value());
      });
  context_future.Force();
}

void ReadTask::Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
  if (!promise_.result_needed()) {
    return;
  }
  // Clear working state.
  crc32c_ = std::nullopt;
  combined_crc32c_ = absl::crc32c_t(0);
  value_.Clear();
  cord_writer_ = riegeli::CordWriter<>(&value_);

  auto context_future = driver_->AllocateContext();
  context_future.ExecuteWhenReady(
      [self = internal::IntrusivePtr<ReadTask>(this),
       context_future](ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
        self->RetryWithContext(std::move(f).value());
      });
  context_future.Force();
}

void ReadTask::RetryWithContext(std::shared_ptr<grpc::ClientContext> context) {
  if (!promise_.result_needed()) {
    return;
  }
  if (g_test_hook) {
    g_test_hook(context.get(), request_);
  }

  storage_generation_ =
      TimestampedStorageGeneration{StorageGeneration::Unknown(), absl::Now()};

  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2))
      << this << " " << ConciseDebugString(request_);

  {
    absl::MutexLock lock(mutex_);
    assert(context_ == nullptr);
    context_ = context;
  }
  auto stub = driver_->get_stub();

  // Start a call.
  intrusive_ptr_increment(this);  // adopted in OnDone.
  stub->async()->ReadObject(context.get(), &request_, this);

  StartRead(&response_);
  StartCall();
}

void ReadTask::OnReadDone(bool ok) {
  if (!ok) return;  // Reading is complete. Not an error.

  if (!promise_.result_needed()) {
    TryCancel();
    return;
  }

  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2))
      << this << " " << ConciseDebugString(response_);

  if (response_.has_checksummed_data()) {
    common_metrics_.bytes_read.IncrementBy(
        response_.checksummed_data().content().size());
  }
  if (auto status = HandleResponse(response_); !status.ok()) {
    promise_.SetResult(status);
    TryCancel();
    return;
  }

  // Issue next request, if necessary.
  StartRead(&response_);
}

void ReadTask::OnDone(const grpc::Status& s) {
  internal::IntrusivePtr<ReadTask> self(this, internal::adopt_object_ref);
  driver_->executor()(
      [self = std::move(self), status = GrpcStatusToAbslStatus(s)]() {
        self->ReadFinished(std::move(status));
      });
}

void ReadTask::ReadFinished(absl::Status status) {
  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2)) << this << " " << status;

  // Streaming read complete.
  if (!promise_.result_needed()) {
    return;
  }

  {
    absl::MutexLock lock(mutex_);
    context_ = nullptr;
  }

  auto latency = absl::Now() - storage_generation_.time;
  common_metrics_.read_latency_ms.Observe(absl::ToInt64Milliseconds(latency));

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
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->Retry(); });
    if (status.ok()) {
      return;
    }
  }

  promise_.SetResult(HandleFinalStatus(status));
}

}  // namespace

void SetTestHook(
    absl::AnyInvocable<void(grpc::ClientContext*,
                            google::storage::v2::ReadObjectRequest&)>&&
        debug_hook) {
  g_test_hook = std::move(debug_hook);
}

/// Implements GcsGrpcKeyValueStore::Read
Future<kvstore::ReadResult> InitiateRead(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    internal_kvstore::CommonMetrics& common_metrics, std::string_view key,
    kvstore::ReadOptions&& options) {
  auto op = PromiseFuturePair<kvstore::ReadResult>::Make();

  auto task = internal::MakeIntrusivePtr<ReadTask>(
      std::move(driver), common_metrics, std::move(options),
      std::move(op.promise));
  task->SetupRequest(task->driver_->bucket_name(), key);
  task->Start();
  return std::move(op.future);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
