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
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/client_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/create_channel.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
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
  internal_metrics::Counter<int64_t>& delete_calls;
  // no additional members
};

auto tsgrpc_metrics = []() -> TsGrpcMetrics {
  return {TENSORSTORE_KVSTORE_COMMON_READ_METRICS(tsgrpc),
          TENSORSTORE_KVSTORE_COMMON_WRITE_METRICS(tsgrpc),
          TENSORSTORE_KVSTORE_COUNTER_IMPL(
              tsgrpc, delete_calls, "kvstore::Write calls deleting a key")};
}();

ABSL_CONST_INIT internal_log::VerboseFlag verbose_logging("tsgrpc_kvstore");

constexpr size_t kMaxWriteChunkSize = 1 << 20;

struct TsGrpcKeyValueStoreSpecData {
  std::string address;
  absl::Duration timeout;
  Context::Resource<GrpcClientCredentials> credentials;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.address, x.timeout, x.credentials, x.data_copy_concurrency);
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

  KvStoreService::StubInterface* stub() { return stub_.get(); }

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

  TsGrpcKeyValueStoreSpecData spec_;
  std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy> auth_strategy_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<KvStoreService::StubInterface> stub_;
};

void MaybeSetDeadline(grpc::ClientContext& context, absl::Duration timeout) {
  if (timeout > absl::ZeroDuration() && timeout != absl::InfiniteDuration()) {
    context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }
}

////////////////////////////////////////////////////

// Implements TsGrpcKeyValueStore::Read
// TODO: Add retries.
struct ReadTask : public internal::AtomicReferenceCount<ReadTask>,
                  public grpc::ClientReadReactor<ReadResponse> {
  Executor executor_;
  Promise<kvstore::ReadResult> promise_;

  // working state.
  std::shared_ptr<grpc::ClientContext> context_;
  kvstore::ReadOptions options_;
  ReadRequest request_;
  ReadResponse response_;
  kvstore::ReadResult result_;

  ReadTask(Executor executor, Promise<kvstore::ReadResult> promise)
      : executor_(std::move(executor)), promise_(std::move(promise)) {}

  void TryCancel() { context_->TryCancel(); }

  void Start(GrpcAuthenticationStrategy& auth_strategy, absl::Duration timeout,
             KvStoreService::StubInterface* stub) {
    context_ = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context_, timeout);
    auto context_future = auth_strategy.ConfigureContext(context_);

    context_future.ExecuteWhenReady(
        [stub, self = internal::IntrusivePtr<ReadTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartImpl(stub);
        });
  }

  void StartImpl(KvStoreService::StubInterface* stub) {
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<ReadTask>(this)] { self->TryCancel(); });

    intrusive_ptr_increment(this);  // adopted in OnDone.
    stub->async()->Read(context_.get(), &request_, this);

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
      promise_.SetResult(std::move(status));
      TryCancel();
    }
  }

  void OnDone(const grpc::Status& s) override {
    internal::IntrusivePtr<ReadTask> self(this, internal::adopt_object_ref);
    executor_([self = std::move(self), status = s]() {
      self->ReadFinished(GrpcStatusToAbslStatus(status));
    });
  }

  void ReadFinished(absl::Status status) {
    // Streaming read complete.
    if (!promise_.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, verbose_logging)
        << "ReadTask::ReadFinished " << ConciseDebugString(response_) << " "
        << status;

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

  auto task =
      internal::MakeIntrusivePtr<ReadTask>(executor(), std::move(pair.promise));
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

  task->Start(*auth_strategy_, spec_.timeout, stub_.get());
  return std::move(pair.future);
}

//////////////////////////////////////////////////////////////////////////

// Implements TsGrpcKeyValueStore::Write
// TODO: Add retries.
struct WriteTask : public internal::AtomicReferenceCount<WriteTask>,
                   public grpc::ClientWriteReactor<WriteRequest> {
  Executor executor_;
  Promise<TimestampedStorageGeneration> promise_;
  absl::Cord value_;

  // working state.
  std::shared_ptr<grpc::ClientContext> context_;
  WriteRequest request_;
  WriteResponse response_;
  size_t value_offset_ = 0;

  WriteTask(Executor executor, Promise<TimestampedStorageGeneration> promise,
            absl::Cord value)
      : executor_(std::move(executor)),
        promise_(std::move(promise)),
        value_(std::move(value)) {}

  void UpdateForNextWrite() {
    auto next_part = value_.Subcord(value_offset_, kMaxWriteChunkSize);
    value_offset_ = std::min(value_.size(), value_offset_ + next_part.size());
    request_.set_value_part(std::move(next_part));
  }

  void TryCancel() { context_->TryCancel(); }

  void Start(GrpcAuthenticationStrategy& auth_strategy, absl::Duration timeout,
             KvStoreService::StubInterface* stub) {
    context_ = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context_, timeout);
    auto context_future = auth_strategy.ConfigureContext(context_);

    context_future.ExecuteWhenReady(
        [stub, self = internal::IntrusivePtr<WriteTask>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartImpl(stub);
        });
  }

  void StartImpl(KvStoreService::StubInterface* stub) {
    promise_.ExecuteWhenNotNeeded([self = internal::IntrusivePtr<WriteTask>(
                                       this)] { self->TryCancel(); });

    intrusive_ptr_increment(this);  // adopted in OnDone.
    stub->async()->Write(context_.get(), &response_, this);

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
    executor_([self = std::move(self), status = s]() {
      self->WriteFinished(GrpcStatusToAbslStatus(status));
    });
  }

  void WriteFinished(absl::Status status) {
    if (!promise_.result_needed()) {
      return;
    }
    ABSL_LOG_IF(INFO, verbose_logging)
        << "WriteTask::WriteFinished " << ConciseDebugString(response_) << " "
        << status;

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
  Executor executor_;
  Promise<TimestampedStorageGeneration> promise_;
  std::shared_ptr<grpc::ClientContext> context_;
  DeleteRequest request_;
  DeleteResponse response_;

  DeleteCallbackState(Executor executor,
                      Promise<TimestampedStorageGeneration> promise)
      : executor_(std::move(executor)), promise_(std::move(promise)) {}

  void TryCancel() { context_->TryCancel(); }

  void Start(GrpcAuthenticationStrategy& auth_strategy, absl::Duration timeout,
             KvStoreService::StubInterface* stub) {
    context_ = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context_, timeout);
    auto context_future = auth_strategy.ConfigureContext(context_);

    context_future.ExecuteWhenReady(
        [stub, self = internal::IntrusivePtr<DeleteCallbackState>(this)](
            ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          self->StartImpl(stub);
        });
  }

  void StartImpl(KvStoreService::StubInterface* stub) {
    promise_.ExecuteWhenNotNeeded(
        [self = internal::IntrusivePtr<DeleteCallbackState>(this)] {
          self->TryCancel();
        });

    stub->async()->Delete(
        context_.get(), &request_, &response_,
        WithExecutor(
            executor_, [self = internal::IntrusivePtr<DeleteCallbackState>(
                            this)](::grpc::Status s) {
              if (!self->promise_.result_needed()) return;
              self->promise_.SetResult(self->Ready(GrpcStatusToAbslStatus(s)));
            }));
  }

  Result<TimestampedStorageGeneration> Ready(absl::Status status) {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "DeleteCallbackState " << ConciseDebugString(response_) << " "
        << status;
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
        executor(), std::move(pair.promise));
    auto& request = task->request_;
    request.set_key(std::move(key));
    request.set_generation_if_equal(
        options.generation_conditions.if_equal.value);

    task->Start(*auth_strategy_, spec_.timeout, stub_.get());
    return std::move(pair.future);
  }

  tsgrpc_metrics.write.Increment();

  auto task = internal::MakeIntrusivePtr<WriteTask>(
      executor(), std::move(pair.promise), *std::move(value));

  auto& request = task->request_;
  request.set_key(std::move(key));
  request.set_generation_if_equal(options.generation_conditions.if_equal.value);

  task->Start(*auth_strategy_, spec_.timeout, stub_.get());
  return std::move(pair.future);
}

Future<const void> TsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  if (range.empty()) return absl::OkStatus();
  tsgrpc_metrics.delete_range.Increment();
  auto pair = PromiseFuturePair<TimestampedStorageGeneration>::Make();

  auto task = internal::MakeIntrusivePtr<DeleteCallbackState>(
      executor(), std::move(pair.promise));
  auto& request = task->request_;
  request.mutable_range()->set_inclusive_min(range.inclusive_min);
  request.mutable_range()->set_exclusive_max(range.exclusive_max);

  task->Start(*auth_strategy_, spec_.timeout, stub_.get());

  return MapFutureValue(
      InlineExecutor{},
      [](auto& f) -> Result<void> { return absl::OkStatus(); },
      std::move(pair.future));
}

// Implements TsGrpcKeyValueStore::List
// NOTE: Convert to async().
struct ListTask : public internal::AtomicReferenceCount<ListTask> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver;
  ListReceiver receiver;

  std::shared_ptr<grpc::ClientContext> context_;
  std::atomic<bool> cancelled = false;
  ListRequest request;

  ListTask(internal::IntrusivePtr<TsGrpcKeyValueStore>&& driver,
           ListReceiver&& receiver)
      : driver(std::move(driver)), receiver(std::move(receiver)) {}

  bool is_cancelled() { return cancelled.load(std::memory_order_relaxed); }

  void try_cancel() {
    if (!cancelled.load()) {
      cancelled.store(true, std::memory_order_relaxed);
      context_->TryCancel();
    }
  }

  void Start() {
    context_ = std::make_shared<grpc::ClientContext>();
    MaybeSetDeadline(*context_, driver->spec_.timeout);

    auto context_future = driver->auth_strategy_->ConfigureContext(context_);
    context_future.ExecuteWhenReady(
        WithExecutor(driver->executor(),
                     [self = internal::IntrusivePtr<ListTask>(this)](
                         ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
                       self->Run();
                     }));
  }

  void Run() {
    // Start a call.
    auto reader = driver->stub()->List(context_.get(), request);

    execution::set_starting(receiver, [this] { try_cancel(); });

    absl::Status msg_status;
    ListResponse response;
    while (reader->Read(&response)) {
      msg_status = GetMessageStatus(response);
      if (!msg_status.ok()) {
        try_cancel();
        break;
      }
      for (const auto& entry : response.entry()) {
        execution::set_value(receiver, ListEntry{entry.key(), entry.size()});
        if (is_cancelled()) break;
      }
      if (is_cancelled()) break;
    }

    auto s = reader->Finish();
    if (!msg_status.ok()) {
      execution::set_error(receiver, msg_status);
    } else if (s.ok() || is_cancelled()) {
      execution::set_done(receiver);
    } else {
      execution::set_error(receiver, GrpcStatusToAbslStatus(s));
    }
    execution::set_stopping(receiver);
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
  auto& request = task->request;
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

  // Create a communication channel with credentials, then use that
  // to construct a gprc stub.
  //
  // TODO: Determine a better mapping to a grpc credentials for this.
  // grpc::Credentials ties the authentication to the communication
  // channel See: <grpcpp/security/credentials.h>,
  // https://grpc.io/docs/guides/auth/
  ABSL_LOG_IF(INFO, verbose_logging)
      << "tsgrpc_kvstore address=" << data_.address;

  driver->auth_strategy_ = data_.credentials->GetAuthenticationStrategy();

  grpc::ChannelArguments args;
  driver->channel_ = internal_grpc::CreateChannel(*driver->auth_strategy_,
                                                  data_.address, args);
  driver->stub_ = KvStoreService::NewStub(driver->channel_);
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
