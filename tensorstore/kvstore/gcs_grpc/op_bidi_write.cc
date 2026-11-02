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

#include "tensorstore/kvstore/gcs_grpc/op_bidi_write.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/impl/call_op_set.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/gcs_grpc/op_delete.h"
#include "tensorstore/kvstore/gcs_grpc/op_move.h"
#include "tensorstore/kvstore/gcs_grpc/test_hook.h"
#include "tensorstore/kvstore/gcs_grpc/utils.h"
#include "tensorstore/kvstore/gcs_grpc/zonal_bucket_state.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

// proto
#include "google/rpc/status.pb.h"
#include "google/storage/v2/storage.pb.h"

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

using ::google::storage::v2::BidiWriteObjectRedirectedError;
using ::google::storage::v2::BidiWriteObjectRequest;
using ::google::storage::v2::BidiWriteObjectResponse;
using ::tensorstore::internal::GrpcStatusToAbslStatus;

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

static constexpr size_t kMaxWriteBytes =
    google::storage::v2::ServiceConstants::MAX_WRITE_CHUNK_BYTES;

// Maximum number of redirects followed by a single write before the redirect
// is surfaced as an error.  Redirects are not counted against the retry
// budget, so this bounds a pathological redirect loop.
static constexpr int kMaxRedirects = 10;

ABSL_CONST_INIT absl::Mutex g_test_hook_mutex(absl::kConstInit);
absl::AnyInvocable<void(grpc::ClientContext*,
                        google::storage::v2::BidiWriteObjectRequest&)>
    g_test_hook ABSL_GUARDED_BY(g_test_hook_mutex);

// Returns the routing token from a `BidiWriteObjectRedirectedError` attached to
// an `ABORTED` status, or `std::nullopt` if `status` is not a redirect.
std::optional<std::string> GetRedirectRoutingToken(const grpc::Status& status) {
  if (status.error_code() != grpc::StatusCode::ABORTED ||
      status.error_details().empty()) {
    return std::nullopt;
  }
  ::google::rpc::Status rpc_status;
  if (!rpc_status.ParseFromString(status.error_details())) {
    return std::nullopt;
  }
  BidiWriteObjectRedirectedError redirect;
  for (const auto& detail : rpc_status.details()) {
    if (detail.Is<BidiWriteObjectRedirectedError>() &&
        detail.UnpackTo(&redirect) && !redirect.routing_token().empty()) {
      return redirect.routing_token();
    }
  }
  return std::nullopt;
}

// Implements a single object upload using BidiWriteObject.
// rpc BidiWriteObject(stream BidiWriteObjectRequest) returns (stream
// BidiWriteObjectResponse) {}
struct BidiWriteTask : public internal::AtomicReferenceCount<BidiWriteTask>,
                       public grpc::ClientBidiReactor<BidiWriteObjectRequest,
                                                      BidiWriteObjectResponse> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  // When non-null, this task is a complete kvstore write and records the
  // write metrics.  Null when the upload is one step of a larger operation,
  // which then records the metrics itself.
  internal_kvstore::CommonMetrics* metrics_;
  std::string bucket_name_;
  std::string object_name_;
  kvstore::WriteOptions options_;
  absl::Cord value_;
  const bool appendable_;
  ZonalProbeLease probe_lease_;
  // If non-empty, stored as `kWriteIdMetadataKey` in the object metadata.
  std::string write_id_;
  Promise<TimestampedStorageGeneration> promise_;
  const absl::Time op_start_time_ = absl::Now();

  // working state.
  size_t value_offset_ = 0;
  absl::crc32c_t crc32c_{0};
  absl::Time start_time_;
  BidiWriteObjectRequest request_;
  BidiWriteObjectResponse response_;
  std::optional<google::storage::v2::Object> final_resource_;
  // Generation of an appendable object created by an earlier attempt on this
  // task (e.g. during an initial `state_lookup`), used to overwrite that exact
  // generation on retry without mutating `options_.generation_conditions`.
  std::optional<uint64_t> created_generation_;
  bool is_first_message_ = true;
  // Routing token from the most recent `BidiWriteObjectRedirectedError`.
  std::string routing_token_;

  // Tracks the lifecycle of a single BidiWriteObject gRPC stream attempt.
  enum class StreamState {
    // No stream is currently active (initial state or between retries).
    kIdle,
    // Sent an initial 0-byte `state_lookup` probe message; both `OnWriteDone`
    // for the 0-byte request and `OnReadDone` for the lookup response are
    // still pending.
    kProbeWriteAndReadPending,
    // `OnWriteDone` completed for the 0-byte `state_lookup` message; waiting
    // for `OnReadDone` before streaming payload chunks.
    kProbeReadPending,
    // `OnReadDone` arrived for the 0-byte `state_lookup` message while
    // `OnWriteDone` is still in flight; payload streaming starts as soon as
    // `OnWriteDone` fires.
    kProbeWritePending,
    // Actively streaming payload chunks (`value_offset_ < value_.size()`);
    // each `OnWriteDone` sends the next chunk.
    kStreaming,
    // The final chunk (`finish_write: true`) has been written; waiting for the
    // final response and `OnDone`.
    kFinishing,
  };

  int redirects_ = 0;
  GrpcCallContext call_;
  absl::Mutex mutex_;
  StreamState stream_state_ ABSL_GUARDED_BY(mutex_) = StreamState::kIdle;

  BidiWriteTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
                internal_kvstore::CommonMetrics* metrics,
                std::string object_name, kvstore::WriteOptions options,
                absl::Cord data, bool appendable, ZonalProbeLease probe_lease,
                std::string write_id,
                Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)),
        metrics_(metrics),
        bucket_name_(driver_->bucket_name()),
        object_name_(std::move(object_name)),
        options_(std::move(options)),
        value_(std::move(data)),
        appendable_(appendable),
        probe_lease_(std::move(probe_lease)),
        write_id_(std::move(write_id)),
        promise_(std::move(promise)),
        call_(write_id_.empty() ? MakeWriteId() : write_id_) {
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<BidiWriteTask>(
                                       this)] { self->call_.TryCancel(); });
  }

  StorageGeneration EffectiveIfEqual() const {
    return created_generation_.has_value()
               ? StorageGeneration::FromUint64(*created_generation_)
               : options_.generation_conditions.if_equal;
  }

  Result<TimestampedStorageGeneration> HandleFinalStatus(absl::Status status);
  void UpdateRequestForNextWrite();
  void StartNextWrite() ABSL_LOCKS_EXCLUDED(mutex_);

  void OnWriteDone(bool ok) override ABSL_LOCKS_EXCLUDED(mutex_);
  void OnReadDone(bool ok) override ABSL_LOCKS_EXCLUDED(mutex_);
  void OnDone(const grpc::Status& s) override;

  void Start() ABSL_LOCKS_EXCLUDED(mutex_);
  void Retry() ABSL_LOCKS_EXCLUDED(mutex_);
  void RetryWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_);
  void OnContextError(absl::Status status) ABSL_LOCKS_EXCLUDED(mutex_) {
    WriteFinished(std::move(status), std::nullopt);
  }
  void WriteFinished(absl::Status status,
                     std::optional<std::string> redirect_routing_token)
      ABSL_LOCKS_EXCLUDED(mutex_);
};

Result<TimestampedStorageGeneration> BidiWriteTask::HandleFinalStatus(
    absl::Status status) {
  TimestampedStorageGeneration result;
  result.time = start_time_;
  if (IsWritePreconditionMismatch(status, EffectiveIfEqual())) {
    // Precondition mismatch.
    //
    // NOTE: If an earlier attempt committed on the server and its response was
    // lost to a retriable error, a backend that honors
    // `x-goog-gcs-idempotency-token` replays the committed `Object` response on
    // retry.  For backends or testbenches that do not replay `BidiWriteObject`
    // by idempotency token, a direct write could recover on `attempt > 0` by
    // stamping `kWriteIdMetadataKey` on the object metadata and issuing a lazy
    // `GetObject` (as `MoveTask::FinishMove` does), verifying that the object
    // carries `write_id_`, has `size() == value_.size()`, and (when
    // `appendable_`) has `has_finalize_time()` so an unfinalized partial object
    // from an earlier attempt is not mistaken for a committed write.
    result.generation = StorageGeneration::Unknown();
    return std::move(result);
  }
  if (!status.ok()) {
    return status;
  }
  if (!final_resource_.has_value()) {
    return absl::InternalError(
        absl::StrFormat("BidiWriteObject completed without resource for %v",
                        tensorstore::QuoteString(object_name_)));
  }
  if (final_resource_->generation() == 0) {
    return absl::InternalError(
        absl::StrFormat("BidiWriteObject completed with zero generation for %v",
                        tensorstore::QuoteString(object_name_)));
  }
  result.generation =
      StorageGeneration::FromUint64(final_resource_->generation());
  return std::move(result);
}

void BidiWriteTask::UpdateRequestForNextWrite() {
  if (is_first_message_) {
    is_first_message_ = false;
  } else {
    // After the first request, clear the spec and any initial state_lookup.
    request_.clear_first_message();
    request_.clear_state_lookup();
  }

  request_.set_write_offset(value_offset_);
  auto& checksummed_data = *request_.mutable_checksummed_data();
  if (value_offset_ == 0 && value_.size() <= kMaxWriteBytes) {
    checksummed_data.set_content(value_);
    value_offset_ = value_.size();
  } else {
    auto next_part = value_.Subcord(value_offset_, kMaxWriteBytes);
    value_offset_ += next_part.size();
    checksummed_data.set_content(std::move(next_part));
  }
  auto chunk_crc32c = ComputeCrc32c(checksummed_data.content());
  checksummed_data.set_crc32c(static_cast<uint32_t>(chunk_crc32c));
  crc32c_ = absl::ConcatCrc32c(crc32c_, chunk_crc32c,
                               checksummed_data.content().size());
  if (value_offset_ == value_.size()) {
    // This is the last request.
    request_.mutable_object_checksums()->set_crc32c(
        static_cast<uint32_t>(crc32c_));
    request_.set_finish_write(true);
  }
}

void BidiWriteTask::StartNextWrite() {
  UpdateRequestForNextWrite();
  auto options = grpc::WriteOptions();
  if (request_.finish_write()) {
    options.set_last_message();
    absl::MutexLock lock(mutex_);
    stream_state_ = StreamState::kFinishing;
  }
  StartWrite(&request_, options);
}

void BidiWriteTask::Start() {
  ABSL_LOG_IF(INFO, gcs_grpc_logging) << "BidiWriteTask " << object_name_;
  Retry();
}

void BidiWriteTask::Retry() {
  if (!promise_.result_needed()) {
    return;
  }

  // Every attempt restarts the upload from offset 0.
  //
  // NOTE: Resuming an interrupted appendable upload instead would require:
  // periodically setting `flush` and `state_lookup` so that the service
  // reports `persisted_size` (it is otherwise only reported when the stream
  // closes); retaining the `generation` and `write_handle` from the responses
  // or from a `BidiWriteObjectRedirectedError`; and reopening the stream with
  // an `AppendObjectSpec` (including the routing token) at `write_offset ==
  // persisted_size`, seeding the running crc32c with the persisted prefix.
  // Appendable uploads here only target short-lived temporary objects, so a
  // full restart is simpler and adequate.
  if (appendable_ && final_resource_.has_value() &&
      final_resource_->generation() != 0 &&
      !StorageGeneration::IsUnknown(options_.generation_conditions.if_equal)) {
    created_generation_ = final_resource_->generation();
  }
  request_.Clear();
  response_.Clear();
  final_resource_.reset();
  start_time_ = absl::Now();
  is_first_message_ = true;
  value_offset_ = 0;
  crc32c_ = absl::crc32c_t{0};

  auto& write_spec = *request_.mutable_write_object_spec();
  auto& resource = *write_spec.mutable_resource();
  resource.set_bucket(bucket_name_);
  resource.set_name(object_name_);
  if (!write_id_.empty()) {
    (*resource.mutable_metadata())[std::string(kWriteIdMetadataKey)] =
        write_id_;
  }
  write_spec.set_appendable(appendable_);
  write_spec.set_object_size(value_.size());
  if (auto gen = EncodeGenerationCondition(EffectiveIfEqual())) {
    write_spec.set_if_generation_match(*gen);
  }

  if (routing_token_.empty()) {
    routing_token_ = driver_->zonal_state().routing_token();
  }

  GrpcCallContext::RunWithContext(internal::IntrusivePtr<BidiWriteTask>(this),
                                  driver_.get(), routing_token_,
                                  &BidiWriteTask::RetryWithContext,
                                  &BidiWriteTask::OnContextError);
}

void BidiWriteTask::RetryWithContext(
    std::shared_ptr<grpc::ClientContext> context) {
  if (!promise_.result_needed() || !call_.BeginAttempt(context)) {
    return;
  }

  // When this task is the single-flight probe resolving the bucket type (or
  // acquiring the initial routing token on a zonal bucket), send a 0-byte
  // `state_lookup` first message and wait for the server's response before
  // computing checksums or streaming the payload.
  const bool send_initial_state_lookup =
      !value_.empty() &&
      (!appendable_
           ? !driver_->zonal_state().is_zonal().has_value()
           : (static_cast<bool>(probe_lease_) && routing_token_.empty()));
  if (send_initial_state_lookup) {
    is_first_message_ = false;
    request_.set_write_offset(0);
    request_.set_state_lookup(true);
  } else {
    UpdateRequestForNextWrite();
  }

  {
    absl::MutexLock lock(g_test_hook_mutex);
    if (g_test_hook) {
      g_test_hook(context.get(), request_);
    }
  }

  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2))
      << this << " write_object_spec={"
      << ConciseDebugString(request_.write_object_spec())
      << "} write_offset=" << request_.write_offset()
      << " bytes=" << request_.checksummed_data().content().size()
      << " state_lookup=" << request_.state_lookup()
      << " finish_write=" << request_.finish_write();

  {
    absl::MutexLock lock(mutex_);
    ABSL_DCHECK(stream_state_ == StreamState::kIdle);
    if (send_initial_state_lookup) {
      stream_state_ = StreamState::kProbeWriteAndReadPending;
    } else if (request_.finish_write()) {
      stream_state_ = StreamState::kFinishing;
    } else {
      stream_state_ = StreamState::kStreaming;
    }
  }
  auto stub = driver_->get_stub();

  // Initiate the write.
  intrusive_ptr_increment(this);
  stub->async()->BidiWriteObject(context.get(), this);

  auto options = grpc::WriteOptions();
  if (request_.finish_write()) {
    options.set_last_message();
  }
  StartWrite(&request_, options);
  StartRead(&response_);
  StartCall();
}

void BidiWriteTask::OnWriteDone(bool ok) {
  if (!ok) return;
  {
    absl::MutexLock lock(mutex_);
    switch (stream_state_) {
      case StreamState::kProbeWriteAndReadPending:
        stream_state_ = StreamState::kProbeReadPending;
        return;
      case StreamState::kProbeWritePending:
        stream_state_ = StreamState::kStreaming;
        break;
      case StreamState::kStreaming:
        break;
      case StreamState::kIdle:
      case StreamState::kProbeReadPending:
      case StreamState::kFinishing:
        return;
    }
  }
  StartNextWrite();
}

void BidiWriteTask::OnReadDone(bool ok) {
  if (!ok) return;
  if (response_.has_resource()) {
    final_resource_ = std::move(*response_.mutable_resource());
  }
  response_.Clear();

  bool resolved_state_lookup = false;
  bool start_next_write = false;
  {
    absl::MutexLock lock(mutex_);
    switch (stream_state_) {
      case StreamState::kProbeWriteAndReadPending:
        stream_state_ = StreamState::kProbeWritePending;
        resolved_state_lookup = true;
        break;
      case StreamState::kProbeReadPending:
        stream_state_ = StreamState::kStreaming;
        resolved_state_lookup = true;
        start_next_write = true;
        break;
      case StreamState::kIdle:
      case StreamState::kProbeWritePending:
      case StreamState::kStreaming:
      case StreamState::kFinishing:
        break;
    }
  }
  if (resolved_state_lookup) {
    if (!appendable_) {
      driver_->zonal_state().set_is_zonal(false);
    } else {
      probe_lease_.Finish(/*mark_done=*/true);
    }
  }
  if (start_next_write) {
    StartNextWrite();
  }
  StartRead(&response_);
}

void BidiWriteTask::OnDone(const grpc::Status& s) {
  internal::IntrusivePtr<BidiWriteTask> self(this, internal::adopt_object_ref);
  driver_->executor()([self = std::move(self),
                       status = GrpcStatusToAbslStatus(s),
                       redirect = GetRedirectRoutingToken(s)]() mutable {
    self->WriteFinished(std::move(status), std::move(redirect));
  });
}

void BidiWriteTask::WriteFinished(
    absl::Status status, std::optional<std::string> redirect_routing_token) {
  ABSL_LOG_IF(INFO, gcs_grpc_logging.Level(2)) << this << " " << status;

  call_.EndAttempt();
  if (!promise_.result_needed()) {
    return;
  }

  {
    absl::MutexLock lock(mutex_);
    stream_state_ = StreamState::kIdle;
  }

  if (redirect_routing_token.has_value() && redirects_ < kMaxRedirects) {
    // The service asked the client to reconnect with a routing token.  This is
    // part of normal operation for zonal buckets, so it is retried immediately
    // and not counted against the retry budget.
    //
    // NOTE: A redirect that also carries a `write_handle` asks the client to
    // resume the partially written generation with an `AppendObjectSpec`.
    // This task does not resume (see `Retry`), so it restarts the upload with
    // the original `WriteObjectSpec`; if the service rejects that, the error
    // is surfaced to the caller.
    ++redirects_;
    routing_token_ = *std::move(redirect_routing_token);
    driver_->zonal_state().set_routing_token(routing_token_);
    Retry();
    return;
  }

  if (!status.ok() && IsRetriable(status) && !routing_token_.empty()) {
    driver_->zonal_state().clear_routing_token(routing_token_);
    routing_token_.clear();
  }
  if (call_.MaybeRetry(driver_.get(), status,
                       [self = internal::IntrusivePtr<BidiWriteTask>(this)] {
                         self->Retry();
                       })) {
    return;
  }

  // Deliberately no in-place appendable/non-appendable transition here.
  // Switching this task to `appendable = true` would write the caller's
  // destination key as an appendable object, which the service makes readable
  // before it is finalized; that is precisely the atomicity violation that
  // `ZonalWriteTask`'s temporary object exists to avoid.  Mode policy is the
  // caller's responsibility -- see `InitiateBidiWrite`.

  auto result = HandleFinalStatus(status);
  if (result.ok()) {
    if (!appendable_) {
      // A non-appendable write was accepted (or rejected only due to a
      // generation precondition mismatch), so the bucket is not zonal.  The
      // converse does not hold: a successful appendable write says nothing,
      // because regional buckets may also support appendable objects.
      driver_->zonal_state().set_is_zonal(false);
    } else {
      probe_lease_.Finish(/*mark_done=*/true);
    }
    if (!StorageGeneration::IsUnknown(result->generation) && metrics_) {
      metrics_->write_latency_ms.Observe(
          absl::ToInt64Milliseconds(absl::Now() - op_start_time_));
      metrics_->bytes_written.IncrementBy(value_.size());
    }
  } else {
    const bool appendable_not_supported =
        appendable_ && IsAppendableNotSupportedError(result.status());
    if (appendable_not_supported) {
      driver_->zonal_state().set_is_zonal(false);
    }
    probe_lease_.Finish(/*mark_done=*/appendable_not_supported ||
                        driver_->zonal_state().is_zonal() == false);
  }
  promise_.SetResult(std::move(result));
}

// Initiates a single BidiWriteObject upload.
Future<TimestampedStorageGeneration> InitiateDirectWrite(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    internal_kvstore::CommonMetrics* metrics, kvstore::Key key,
    kvstore::Value data, kvstore::WriteOptions options, bool appendable,
    ZonalProbeLease probe_lease, std::string write_id) {
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();
  auto task = internal::MakeIntrusivePtr<BidiWriteTask>(
      std::move(driver), metrics, std::move(key), std::move(options),
      std::move(data), appendable, std::move(probe_lease), std::move(write_id),
      std::move(op.promise));
  task->Start();
  return std::move(op.future);
}

// Generates a temporary object name for appendable writes.
//
// GCS object names are limited to 1024 bytes of valid UTF-8.  Truncate `key`
// on a UTF-8 code-point boundary before appending the 39-byte
// `.__tmp_<128-bit hex>` suffix (`write_id`) so the temporary object name never
// exceeds 1024 bytes.
std::string MakeTempKey(std::string_view key, std::string_view write_id) {
  static constexpr size_t kMaxObjectNameBytes = 1024;
  static constexpr size_t kTempPrefixBytes = 7;  // ".__tmp_"
  const size_t max_key_prefix_bytes =
      kMaxObjectNameBytes - kTempPrefixBytes - write_id.size();
  if (key.size() > max_key_prefix_bytes) {
    size_t len = max_key_prefix_bytes;
    while (len > 0 && (static_cast<uint8_t>(key[len]) & 0xC0) == 0x80) {
      --len;
    }
    key = key.substr(0, len);
  }
  return absl::StrFormat("%s.__tmp_%s", key, write_id);
}

// Coordinates an atomic zonal write (or fallback to zonal write).
//
// Zonal buckets require appendable objects, which are readable before they
// are finalized. To preserve single-key write atomicity, `ZonalWriteTask`
// writes to a temporary object and atomically renames it to the target key
// via `MoveObject`. In fallback mode (when bucket type is unknown), it
// attempts a direct non-appendable write first, and on
// `IsZonalObjectRequiredError`, updates the driver and falls back to the
// temporary object + rename path.
//
// Temporary objects are visible to `List` while a write is in flight.
//
// Cancellation: once the temporary upload has started, it and the subsequent
// `MoveObject` always run to completion, even if the caller cancels the
// write.  Cancelling an upload mid-stream would leave an unfinalized
// appendable object behind whose deletion could race the in-flight request;
// waiting for completion means the exact temporary generation is known and
// can be deleted with a generation precondition.  The cost is that a
// cancelled write still uploads its data, and a cancelled write whose move
// already started may still take effect.
struct ZonalWriteTask : public internal::AtomicReferenceCount<ZonalWriteTask> {
  internal::IntrusivePtr<GcsGrpcKeyValueStore> driver_;
  internal_kvstore::CommonMetrics& metrics_;
  kvstore::Key key_;
  std::string write_id_;
  std::string temp_key_;
  kvstore::Value data_;
  const size_t data_size_;
  kvstore::WriteOptions options_;
  Promise<TimestampedStorageGeneration> promise_;
  ZonalProbeLease probe_lease_;
  StorageGeneration temp_generation_;
  const absl::Time op_start_time_ = absl::Now();

  ZonalWriteTask(internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
                 internal_kvstore::CommonMetrics& metrics, kvstore::Key key,
                 kvstore::Value data, kvstore::WriteOptions options,
                 Promise<TimestampedStorageGeneration> promise,
                 ZonalProbeLease probe_lease)
      : driver_(std::move(driver)),
        metrics_(metrics),
        key_(std::move(key)),
        write_id_(MakeWriteId()),
        temp_key_(MakeTempKey(key_, write_id_)),
        data_(std::move(data)),
        data_size_(data_.size()),
        options_(std::move(options)),
        promise_(std::move(promise)),
        probe_lease_(std::move(probe_lease)) {}

  void Start(std::optional<bool> is_zonal) {
    if (is_zonal == true) {
      StartTempWrite();
    } else {
      StartFallbackWrite();
    }
  }

  // Records the metrics for a successful write.
  void RecordSuccess() {
    metrics_.write_latency_ms.Observe(
        absl::ToInt64Milliseconds(absl::Now() - op_start_time_));
    metrics_.bytes_written.IncrementBy(data_size_);
  }

  void StartFallbackWrite() {
    // No temporary object is involved, so cancellation may propagate to the
    // direct write.  `probe_lease_` remains owned by `ZonalWriteTask` so that
    // if the fallback write fails with `IsZonalObjectRequiredError`, the same
    // lease can be transferred to `StartTempWrite()`.
    auto direct_future = InitiateDirectWrite(
        driver_, /*metrics=*/nullptr, key_, data_, options_,
        /*appendable=*/false, /*probe_lease=*/{}, /*write_id=*/{});
    Link(
        [self = internal::IntrusivePtr<ZonalWriteTask>(this)](
            Promise<TimestampedStorageGeneration> promise,
            ReadyFuture<TimestampedStorageGeneration> f) {
          self->OnFallbackDirectWriteDone(f.result());
        },
        promise_, direct_future);
  }

  void OnFallbackDirectWriteDone(
      const Result<TimestampedStorageGeneration>& result) {
    if (!result.ok() && IsZonalObjectRequiredError(result.status())) {
      driver_->zonal_state().set_is_zonal(true);
      if (driver_->zonal_state().is_zonal() == true) {
        StartTempWrite();
        return;
      }
      // Contradicts an earlier signal; report the error rather than looping.
    }
    probe_lease_.Finish(
        /*mark_done=*/driver_->zonal_state().is_zonal().has_value());
    if (result.ok() && !StorageGeneration::IsUnknown(result->generation)) {
      RecordSuccess();
    }
    promise_.SetResult(result);
  }

  void StartTempWrite() {
    // Not linked to `promise_`; see the cancellation note above.
    // `if_equal` is intentionally left unset (`Unknown()`): the 128-bit random
    // suffix in `temp_key_` avoids collisions with concurrent writers, while an
    // unconditional write allows `BidiWriteTask::Retry` to overwrite a
    // partially written appendable object from offset 0 if a stream fails after
    // the initial `WriteObjectSpec` has already created the object.
    auto write_future = InitiateDirectWrite(
        driver_, /*metrics=*/nullptr, temp_key_, std::move(data_),
        kvstore::WriteOptions{}, /*appendable=*/true, std::move(probe_lease_),
        write_id_);
    write_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ZonalWriteTask>(this)](
            ReadyFuture<TimestampedStorageGeneration> f) {
          self->OnTempWriteDone(f.result());
        });
  }

  void OnTempWriteDone(const Result<TimestampedStorageGeneration>& result) {
    if (!result.ok()) {
      // The upload has completed, so no request can still create the object;
      // remove any partially written appendable object unconditionally.
      CleanupAndSetResult(StorageGeneration::Unknown(), result.status());
      return;
    }
    ABSL_DCHECK(!StorageGeneration::IsUnknown(result->generation));
    temp_generation_ = result->generation;
    if (!promise_.result_needed()) {
      // Cancelled while uploading.
      CleanupAndSetResult(temp_generation_, absl::CancelledError());
      return;
    }
    StartMove();
  }

  void StartMove() {
    // Not linked to `promise_`; see the cancellation note above.
    auto move_future = InitiateMoveObject(driver_, temp_key_, key_, options_,
                                          temp_generation_, write_id_);
    move_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ZonalWriteTask>(this)](
            ReadyFuture<TimestampedStorageGeneration> f) {
          self->OnMoveDone(f.result());
        });
  }

  void OnMoveDone(const Result<TimestampedStorageGeneration>& result) {
    if (result.ok() && !StorageGeneration::IsUnknown(result->generation)) {
      // The move consumed the temporary object.
      RecordSuccess();
      promise_.SetResult(result);
      return;
    }
    CleanupAndSetResult(temp_generation_, result);
  }

  // Deletes the temporary object, then completes the write with `result`.
  void CleanupAndSetResult(StorageGeneration temp_generation,
                           Result<TimestampedStorageGeneration> result) {
    kvstore::WriteOptions options;
    options.generation_conditions.if_equal = std::move(temp_generation);
    InitiateDelete(driver_, temp_key_, std::move(options))
        .ExecuteWhenReady([self = internal::IntrusivePtr<ZonalWriteTask>(this),
                           result = std::move(result)](
                              ReadyFuture<TimestampedStorageGeneration> f) {
          if (!f.result().ok()) {
            ABSL_LOG_IF(INFO, gcs_grpc_logging)
                << "Failed to clean up temporary object "
                << tensorstore::QuoteString(self->temp_key_) << " in "
                << self->driver_->bucket_name() << ": " << f.result().status();
          }
          self->promise_.SetResult(result);
        });
  }
};

}  // namespace

bool IsZonalObjectRequiredError(const absl::Status& status) {
  if (status.code() != absl::StatusCode::kInvalidArgument) {
    return false;
  }
  const std::string_view msg = status.message();
  return absl::StrContains(msg, "bucket requires appendable objects") ||
         absl::StrContains(msg, "bucket type only supports appendable objects");
}

bool IsAppendableNotSupportedError(const absl::Status& status) {
  if (status.code() != absl::StatusCode::kUnimplemented &&
      status.code() != absl::StatusCode::kInvalidArgument) {
    return false;
  }
  return absl::StrContains(status.message(),
                           "Appendable objects are not supported");
}

Future<TimestampedStorageGeneration> InitiateBidiWrite(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    internal_kvstore::CommonMetrics& common_metrics, kvstore::Key key,
    kvstore::Value data, kvstore::WriteOptions options) {
  auto probe_result = driver->zonal_state().AcquireProbe();
  if (probe_result.action == ZonalBucketState::ProbeAction::kWait) {
    return PromiseFuturePair<TimestampedStorageGeneration>::Link(
               [driver = std::move(driver), &common_metrics,
                key = std::move(key), data = std::move(data),
                options = std::move(options)](
                   Promise<TimestampedStorageGeneration> promise,
                   ReadyFuture<const void> /*ready_future*/) mutable {
                 if (!promise.result_needed()) return;
                 LinkResult(std::move(promise),
                            InitiateBidiWrite(std::move(driver), common_metrics,
                                              std::move(key), std::move(data),
                                              std::move(options)));
               },
               std::move(probe_result.wait_future))
        .future;
  }

  const std::optional<bool> is_zonal = driver->zonal_state().is_zonal();
  if (is_zonal == false) {
    // Non-appendable write: write and fail, just like a normal write.
    return InitiateDirectWrite(std::move(driver), &common_metrics,
                               std::move(key), std::move(data),
                               std::move(options), /*appendable=*/false,
                               /*probe_lease=*/{}, /*write_id=*/{});
  }
  if (driver->zonal_state().is_non_atomic()) {
    // Non-atomic zonal write: write directly to `key` as an appendable object
    // without a temporary object or MoveObject rename.
    return InitiateDirectWrite(std::move(driver), &common_metrics,
                               std::move(key), std::move(data),
                               std::move(options), /*appendable=*/true,
                               std::move(probe_result.lease), /*write_id=*/{});
  }

  // Possibly appendable write.
  auto op = PromiseFuturePair<TimestampedStorageGeneration>::Make();
  auto task = internal::MakeIntrusivePtr<ZonalWriteTask>(
      std::move(driver), common_metrics, std::move(key), std::move(data),
      std::move(options), std::move(op.promise), std::move(probe_result.lease));
  task->Start(is_zonal);
  return std::move(op.future);
}

void SetTestHook(
    absl::AnyInvocable<void(grpc::ClientContext*,
                            google::storage::v2::BidiWriteObjectRequest&)>&&
        debug_hook) {
  absl::MutexLock lock(g_test_hook_mutex);
  g_test_hook = std::move(debug_hook);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
