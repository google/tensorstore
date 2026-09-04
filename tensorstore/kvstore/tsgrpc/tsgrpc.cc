// Copyright 2021 The TensorStore Authors
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

/// \file
/// Key-value store proxied over grpc.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/impl/call_op_set.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/grpc/channel_options.h"
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/stub_pool.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/thread/schedule_at.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/tsgrpc/common.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep

// protos
#include "tensorstore/kvstore/tsgrpc/common.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.pb.h"

using ::tensorstore::GrpcClientCredentials;
using ::tensorstore::internal::AbslTimeToProto;
using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::GrpcStatusToAbslStatus;
using ::tensorstore::internal_grpc::GrpcAuthenticationStrategy;
using ::tensorstore::internal_grpc::IsRetriable;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore_grpc::DecodeGenerationAndTimestamp;
using ::tensorstore_grpc::GetMessageStatus;
using ::tensorstore_grpc::kvstore::DeleteRequest;
using ::tensorstore_grpc::kvstore::DeleteResponse;
using ::tensorstore_grpc::kvstore::ListRequest;
using ::tensorstore_grpc::kvstore::ListResponse;
using ::tensorstore_grpc::kvstore::ReadRequest;
using ::tensorstore_grpc::kvstore::ReadResponse;
using ::tensorstore_grpc::kvstore::WriteRequest;
using ::tensorstore_grpc::kvstore::WriteResponse;
using ::tensorstore_grpc::kvstore::grpc_gen::KvStoreService;

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

struct TsGrpcMetrics : public internal_kvstore::CommonReadMetrics,
                       public internal_kvstore::CommonWriteMetrics {
  internal_metrics::Counter<int64_t> delete_calls;
  internal_metrics::Counter<int64_t> retries;
};
ABSL_CONST_INIT static TsGrpcMetrics tsgrpc_metrics;

TENSORSTORE_GLOBAL_INITIALIZER {
  TENSORSTORE_KVSTORE_REGISTER_COMMON_READ_METRICS(&tsgrpc_metrics, tsgrpc);
  TENSORSTORE_KVSTORE_REGISTER_COMMON_WRITE_METRICS(&tsgrpc_metrics, tsgrpc);
  auto& r = internal_metrics::GetMetricRegistry();
  r.Register(&tsgrpc_metrics.delete_calls,
             internal_metrics::MetricMetadata(
                 "/tensorstore/kvstore/tsgrpc/delete_calls",
                 "tsgrpc kvstore::Write calls deleting a key"));
  r.Register(&tsgrpc_metrics.retries,
             internal_metrics::MetricMetadata(
                 "/tensorstore/kvstore/tsgrpc/retries", "tsgrpc retries"));
}

ABSL_CONST_INIT internal_log::VerboseFlag verbose_logging("tsgrpc_kvstore");

constexpr size_t kMaxWriteChunkSize = 1 << 20;

/// Specifies a limit on the number of retries.
struct TsGrpcRequestRetries
    : public internal::RetriesResource<TsGrpcRequestRetries> {
  static constexpr char id[] = "tsgrpc_request_retries";
};

const internal::ContextResourceRegistration<TsGrpcRequestRetries>
    tsgrpc_request_retries_registration;

struct TsGrpcKeyValueStoreSpecData {
  std::string address;
  absl::Duration timeout;
  Context::Resource<TsGrpcRequestRetries> retries;
  Context::Resource<GrpcClientCredentials> credentials;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.address, x.timeout, x.retries, x.credentials,
             x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member(GrpcClientCredentials::id,
                 jb::Projection<&TsGrpcKeyValueStoreSpecData::credentials>()),
      jb::Member("address",
                 jb::Projection<&TsGrpcKeyValueStoreSpecData::address>()),
      jb::Member("timeout",
                 jb::Projection<&TsGrpcKeyValueStoreSpecData::timeout>(
                     jb::DefaultValue<jb::kNeverIncludeDefaults>(
                         [](auto* x) { *x = absl::Seconds(60); }))),
      jb::Member(TsGrpcRequestRetries::id,
                 jb::Projection<&TsGrpcKeyValueStoreSpecData::retries>()),
      jb::Member(
          DataCopyConcurrencyResource::id,
          jb::Projection<
              &TsGrpcKeyValueStoreSpecData::data_copy_concurrency>()) /**/
  );
};

class TsGrpcKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          TsGrpcKeyValueStoreSpec, TsGrpcKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "tsgrpc_kvstore";
  Future<kvstore::DriverPtr> DoOpen() const override;
};

/// Defines the "tsgrpc_kvstore" KeyValueStore driver.
class TsGrpcKeyValueStore
    : public internal_kvstore::RegisteredDriver<TsGrpcKeyValueStore,
                                                TsGrpcKeyValueStoreSpec> {
 public:
  TsGrpcKeyValueStore(const TsGrpcKeyValueStoreSpecData& spec) : spec_(spec) {}

  const Executor& executor() const {
    return spec_.data_copy_concurrency->executor;
  }

  std::shared_ptr<KvStoreService::StubInterface> stub() const {
    return stub_pool_->get_next_stub();
  }

  /// Obtains a `SpecData` representation from an open `Driver`.
  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<const void> DeleteRange(KeyRange range) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  absl::Status BackoffForAttemptAsync(
      absl::Status status, int attempt, absl::AnyInvocable<void() &&> task,
      SourceLocation loc = SourceLocation::current());

  TsGrpcKeyValueStoreSpecData spec_;
  std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy_;
  std::shared_ptr<internal_grpc::StubPool<KvStoreService::StubInterface>>
      stub_pool_;
};

void MaybeSetDeadline(grpc::ClientContext& context, absl::Duration timeout) {
  if (timeout > absl::ZeroDuration() && timeout != absl::InfiniteDuration()) {
    context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }
}

absl::Status TsGrpcKeyValueStore::BackoffForAttemptAsync(
    absl::Status status, int attempt, absl::AnyInvocable<void() &&> task,
    SourceLocation loc) {
  auto delay = spec_.retries->BackoffForAttempt(attempt);
  if (!delay) {
    return StatusBuilder(std::move(status), loc)
        .SetCode(absl::StatusCode::kAborted)
        .Format("All %d retry attempts failed", spec_.retries->max_retries);
  }
  tsgrpc_metrics.retries.Increment();
  internal::ScheduleAt(
      absl::Now() + *delay,
      WithExecutor(executor(),
                   [task = std::move(task)]() mutable { std::move(task)(); }));
  return absl::OkStatus();
}

////////////////////////////////////////////////////

// Implements TsGrpcKeyValueStore::Read
struct ReadTask : public internal::AtomicReferenceCount<ReadTask>,
                  public grpc::ClientReadReactor<ReadResponse> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver_;
  Promise<kvstore::ReadResult> promise_;

  // working state.
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  absl::Status message_status_ ABSL_GUARDED_BY(mutex_);
  int attempt_ = 0;
  ReadRequest request_;
  ReadResponse response_;
  kvstore::ReadResult result_;

  ReadTask(internal::IntrusivePtr<TsGrpcKeyValueStore> driver,
           Promise<kvstore::ReadResult> promise)
      : driver_(std::move(driver)), promise_(std::move(promise)) {
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->TryCancel(); });
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  void Start() ABSL_LOCKS_EXCLUDED(mutex_) {
    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    result_ = {};
    response_.Clear();

    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void StartWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    auto* context_ptr = context.get();
    {
      absl::MutexLock lock(mutex_);
      context_ = std::move(context);
    }

    intrusive_ptr_increment(this);  // adopted in OnDone.
    driver_->stub()->async()->Read(context_ptr, &request_, this);

    StartRead(&response_);
    StartCall();
  }

  void OnReadDone(bool ok) override {
    if (!ok) return;
    if (!promise_.result_needed()) {
      TryCancel();
      return;
    }

    auto status = [&]() -> absl::Status {
      if (auto status = GetMessageStatus(response_); !status.ok()) {
        return status;
      }

      if (result_.value.empty()) {
        auto stamp = DecodeGenerationAndTimestamp(response_);
        if (!stamp.ok()) {
          return std::move(stamp).status();
        }
        result_.stamp = std::move(stamp).value();
        result_.state =
            static_cast<kvstore::ReadResult::State>(response_.state());
      }

      result_.value.Append(response_.value_part());
      StartRead(&response_);
      return absl::OkStatus();
    }();

    if (!status.ok()) {
      {
        absl::MutexLock lock(mutex_);
        message_status_ = status;
      }
      TryCancel();
    }
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<ReadTask> self(this, internal::adopt_object_ref);
    driver_->executor()([self = std::move(self), status = s]() {
      self->ReadFinished(GrpcStatusToAbslStatus(status));
    });
  }

  void ReadFinished(absl::Status status) {
    // Streaming read complete.
    if (!promise_.result_needed()) {
      return;
    }
    {
      absl::MutexLock lock(mutex_);
      context_ = nullptr;
      if (!message_status_.ok()) {
        status = std::move(message_status_);
        message_status_ = absl::OkStatus();
      }
    }
    ABSL_LOG_IF(INFO, verbose_logging)
        << "ReadTask::ReadFinished " << ConciseDebugString(response_) << " "
        << status;

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

    if (!status.ok()) {
      promise_.SetResult(status);
    } else {
      promise_.SetResult(std::move(result_));
    }
  }
};

/// Key value store operations.
Future<kvstore::ReadResult> TsGrpcKeyValueStore::Read(Key key,
                                                      ReadOptions options) {
  tsgrpc_metrics.read.Increment();

  auto pair = PromiseFuturePair<kvstore::ReadResult>::Make();

  auto task = internal::MakeIntrusivePtr<ReadTask>(
      internal::IntrusivePtr<TsGrpcKeyValueStore>(this),
      std::move(pair.promise));
  auto& request = task->request_;
  request.set_key(std::move(key));
  request.set_generation_if_equal(options.generation_conditions.if_equal.value);
  request.set_generation_if_not_equal(
      options.generation_conditions.if_not_equal.value);
  if (!options.byte_range.IsFull()) {
    request.mutable_byte_range()->set_inclusive_min(
        options.byte_range.inclusive_min);
    request.mutable_byte_range()->set_exclusive_max(
        options.byte_range.exclusive_max);
  }
  if (options.staleness_bound != absl::InfiniteFuture()) {
    AbslTimeToProto(options.staleness_bound, request.mutable_staleness_bound());
  }

  task->Start();
  return std::move(pair.future);
}

//////////////////////////////////////////////////////////////////////////

// Implements TsGrpcKeyValueStore::Write
struct WriteTask : public internal::AtomicReferenceCount<WriteTask>,
                   public grpc::ClientWriteReactor<WriteRequest> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver_;
  Promise<TimestampedStorageGeneration> promise_;
  absl::Cord value_;

  // working state.
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  WriteRequest request_;
  WriteResponse response_;
  size_t value_offset_ = 0;
  int attempt_ = 0;

  WriteTask(internal::IntrusivePtr<TsGrpcKeyValueStore> driver,
            Promise<TimestampedStorageGeneration> promise, absl::Cord value)
      : driver_(std::move(driver)),
        promise_(std::move(promise)),
        value_(std::move(value)) {
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<WriteTask>(
                                       this)] { self->TryCancel(); });
  }

  void UpdateForNextWrite() {
    auto next_part = value_.Subcord(value_offset_, kMaxWriteChunkSize);
    value_offset_ = std::min(value_.size(), value_offset_ + next_part.size());
    request_.set_value_part(std::move(next_part));
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  void Start() ABSL_LOCKS_EXCLUDED(mutex_) {
    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<WriteTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    value_offset_ = 0;
    response_.Clear();

    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<WriteTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void StartWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) {
      return;
    }
    auto* context_ptr = context.get();
    {
      absl::MutexLock lock(mutex_);
      context_ = std::move(context);
    }

    intrusive_ptr_increment(this);  // adopted in OnDone.
    driver_->stub()->async()->Write(context_ptr, &response_, this);

    UpdateForNextWrite();

    auto options = grpc::WriteOptions();
    if (value_offset_ == value_.size()) {
      options.set_last_message();
    }
    StartWrite(&request_, options);
    StartCall();
  }

  void OnWriteDone(bool ok) override {
    // Not streaming any additional data bits.
    if (!ok) return;
    if (value_offset_ < value_.size()) {
      UpdateForNextWrite();

      auto options = grpc::WriteOptions();
      if (value_offset_ == value_.size()) {
        options.set_last_message();
      }
      StartWrite(&request_, options);
    }
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<WriteTask> self(this, internal::adopt_object_ref);
    driver_->executor()([self = std::move(self), status = s]() {
      self->WriteFinished(GrpcStatusToAbslStatus(status));
    });
  }

  void WriteFinished(absl::Status status) {
    if (!promise_.result_needed()) {
      return;
    }
    {
      absl::MutexLock lock(mutex_);
      context_ = nullptr;
    }
    ABSL_LOG_IF(INFO, verbose_logging)
        << "WriteTask::WriteFinished " << ConciseDebugString(response_) << " "
        << status;

    if (!status.ok() && attempt_ == 0 &&
        status.code() == absl::StatusCode::kUnauthenticated) {
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

    promise_.SetResult([&]() -> Result<TimestampedStorageGeneration> {
      TENSORSTORE_RETURN_IF_ERROR(status);
      TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response_));
      return DecodeGenerationAndTimestamp(response_);
    }());
  }
};

//////////////////////////////////////////////////////////////////////////

struct DeleteCallbackState
    : public internal::AtomicReferenceCount<DeleteCallbackState> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver_;
  Promise<TimestampedStorageGeneration> promise_;
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  int attempt_ = 0;
  DeleteRequest request_;
  DeleteResponse response_;

  DeleteCallbackState(internal::IntrusivePtr<TsGrpcKeyValueStore> driver,
                      Promise<TimestampedStorageGeneration> promise)
      : driver_(std::move(driver)), promise_(std::move(promise)) {
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<DeleteCallbackState>(this)] {
          self->TryCancel();
        });
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    if (context_) context_->TryCancel();
  }

  void Start() ABSL_LOCKS_EXCLUDED(mutex_) {
    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<DeleteCallbackState>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void Retry() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) return;
    response_.Clear();

    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);
    auto context_future = driver_->auth_strategy_->ConfigureContext(context);

    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<DeleteCallbackState>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void StartWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!promise_.result_needed()) return;
    auto* context_ptr = context.get();
    {
      absl::MutexLock lock(mutex_);
      context_ = std::move(context);
    }

    driver_->stub()->async()->Delete(
        context_ptr, &request_, &response_,
        WithExecutor(driver_->executor(),
                     [self = internal::IntrusivePtr<DeleteCallbackState>(this)](
                         ::grpc::Status s) { self->OnDone(s); }));
  }

  void OnDone(const ::grpc::Status& s) {
    if (!promise_.result_needed()) return;
    auto status = GrpcStatusToAbslStatus(s);
    {
      absl::MutexLock lock(mutex_);
      context_ = nullptr;
    }
    ABSL_LOG_IF(INFO, verbose_logging)
        << "DeleteCallbackState " << ConciseDebugString(response_) << " "
        << status;

    if (!status.ok() && attempt_ == 0 &&
        status.code() == absl::StatusCode::kUnauthenticated) {
      attempt_++;
      Retry();
      return;
    }
    if (!status.ok() && IsRetriable(status)) {
      status = driver_->BackoffForAttemptAsync(
          std::move(status), attempt_++,
          [self = internal::IntrusivePtr<DeleteCallbackState>(this)] {
            self->Retry();
          });
      if (status.ok()) {
        return;
      }
    }

    promise_.SetResult(Ready(status));
  }

  Result<TimestampedStorageGeneration> Ready(absl::Status status) {
    TENSORSTORE_RETURN_IF_ERROR(status);
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response_));
    return DecodeGenerationAndTimestamp(response_);
  }
};

Future<TimestampedStorageGeneration> TsGrpcKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  auto pair = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  if (!value) {
    // empty value is delete.
    tsgrpc_metrics.delete_calls.Increment();

    auto task = internal::MakeIntrusivePtr<DeleteCallbackState>(
        internal::IntrusivePtr<TsGrpcKeyValueStore>(this),
        std::move(pair.promise));
    auto& request = task->request_;
    request.set_key(std::move(key));
    request.set_generation_if_equal(
        options.generation_conditions.if_equal.value);

    task->Start();
    return std::move(pair.future);
  }

  tsgrpc_metrics.write.Increment();

  auto task = internal::MakeIntrusivePtr<WriteTask>(
      internal::IntrusivePtr<TsGrpcKeyValueStore>(this),
      std::move(pair.promise), *std::move(value));

  auto& request = task->request_;
  request.set_key(std::move(key));
  request.set_generation_if_equal(options.generation_conditions.if_equal.value);

  task->Start();
  return std::move(pair.future);
}

Future<const void> TsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  if (range.empty()) return absl::OkStatus();
  tsgrpc_metrics.delete_range.Increment();
  auto pair = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  auto task = internal::MakeIntrusivePtr<DeleteCallbackState>(
      internal::IntrusivePtr<TsGrpcKeyValueStore>(this),
      std::move(pair.promise));
  auto& request = task->request_;
  request.mutable_range()->set_inclusive_min(range.inclusive_min);
  request.mutable_range()->set_exclusive_max(range.exclusive_max);

  task->Start();

  return MapFutureValue(
      InlineExecutor{},
      [](auto& f) -> Result<void> { return absl::OkStatus(); },
      std::move(pair.future));
}

// Implements TsGrpcKeyValueStore::List
struct ListTask : public internal::AtomicReferenceCount<ListTask>,
                  public grpc::ClientReadReactor<ListResponse> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver_;
  ListReceiver receiver_;

  // Stub must outlive the async call; async() returns a
  // pointer tied to the stub's lifetime.
  std::shared_ptr<KvStoreService::StubInterface> stub_;
  absl::Mutex mutex_;
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  ListRequest request_;
  ListResponse response_;
  absl::Status message_status_ ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> cancelled_ = false;

  ListTask(internal::IntrusivePtr<TsGrpcKeyValueStore> driver,
           ListReceiver receiver)
      : driver_(std::move(driver)), receiver_(std::move(receiver)) {
    execution::set_starting(receiver_, [this] { TryCancel(); });
  }

  ~ListTask() {
    {
      absl::MutexLock lock(mutex_);
      context_ = nullptr;
    }
    driver_ = {};
    execution::set_stopping(receiver_);
  }

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    if (!cancelled_.exchange(true, std::memory_order_relaxed)) {
      absl::MutexLock lock(mutex_);
      if (context_) context_->TryCancel();
    }
  }

  void Start() ABSL_LOCKS_EXCLUDED(mutex_) {
    auto context = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context, driver_->spec_.timeout);

    auto context_future = driver_->auth_strategy_->ConfigureContext(context);
    context_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ListTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartWithContext(std::move(f).value());
        });
  }

  void StartWithContext(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_) {
    if (cancelled_.load(std::memory_order_relaxed)) {
      execution::set_done(receiver_);
      return;
    }

    stub_ = driver_->stub();
    auto* context_ptr = context.get();
    {
      absl::MutexLock lock(mutex_);
      context_ = std::move(context);
    }

    intrusive_ptr_increment(this);  // adopted in OnDone.
    stub_->async()->List(context_ptr, &request_, this);
    StartRead(&response_);
    StartCall();
  }

  void OnReadDone(bool ok) override {
    if (!ok) return;
    if (cancelled_.load(std::memory_order_relaxed)) {
      TryCancel();
      return;
    }

    auto status = GetMessageStatus(response_);
    if (!status.ok()) {
      {
        absl::MutexLock lock(mutex_);
        message_status_ = status;
      }
      TryCancel();
      return;
    }

    for (const auto& entry : response_.entry()) {
      execution::set_value(receiver_, ListEntry{entry.key(), entry.size()});
      if (cancelled_.load(std::memory_order_relaxed)) {
        TryCancel();
        return;
      }
    }
    StartRead(&response_);
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<ListTask> self(this, internal::adopt_object_ref);
    driver_->executor()([self = std::move(self), status = s]() {
      self->ListFinished(GrpcStatusToAbslStatus(status));
    });
  }

  void ListFinished(absl::Status status) {
    {
      absl::MutexLock lock(mutex_);
      if (!message_status_.ok()) {
        status = std::move(message_status_);
      }
    }
    if (cancelled_.load(std::memory_order_relaxed) || status.ok()) {
      execution::set_done(receiver_);
    } else {
      execution::set_error(receiver_, status);
    }
  }
};

void TsGrpcKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }
  tsgrpc_metrics.list.Increment();
  auto task = internal::MakeIntrusivePtr<ListTask>(
      internal::IntrusivePtr<TsGrpcKeyValueStore>(this), std::move(receiver));
  auto& request = task->request_;
  request.mutable_range()->set_inclusive_min(options.range.inclusive_min);
  request.mutable_range()->set_exclusive_max(options.range.exclusive_max);
  request.set_strip_prefix_length(options.strip_prefix_length);
  if (options.staleness_bound != absl::InfiniteFuture()) {
    AbslTimeToProto(options.staleness_bound, request.mutable_staleness_bound());
  }

  task->Start();
}

Future<kvstore::DriverPtr> TsGrpcKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<TsGrpcKeyValueStore>(data_);

  ABSL_LOG_IF(INFO, verbose_logging)
      << "tsgrpc_kvstore address=" << data_.address;

  // TODO: Determine a better mapping to grpc credentials.
  // grpc::Credentials ties the authentication to the channel.
  // See: <grpcpp/security/credentials.h>,
  // https://grpc.io/docs/guides/auth/
  driver->auth_strategy_ = data_.credentials->GetAuthenticationStrategy();
  driver->stub_pool_ =
      internal_grpc::CreateStubPool<KvStoreService,
                                    KvStoreService::StubInterface>(
          data_.address, 0, *driver->auth_strategy_, absl::ZeroDuration());
  return driver;
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::TsGrpcKeyValueStore)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::TsGrpcKeyValueStoreSpec>
    registration;
}
