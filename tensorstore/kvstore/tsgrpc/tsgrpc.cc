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

#include <stdint.h>

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
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "grpcpp/support/sync_stream.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/kvstore/byte_range.h"
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

auto& grpc_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc/read", "grpc driver kvstore::Read calls");

auto& grpc_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc/write", "grpc driver kvstore::Write calls");

auto& grpc_delete = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc/delete", "grpc driver kvstore::Write calls");

auto& grpc_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc/delete_range",
    "grpc driver kvstore::DeleteRange calls");

auto& grpc_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc/list", "grpc driver kvstore::List calls");

ABSL_CONST_INIT internal_log::VerboseFlag verbose_logging("tsgrpc_kvstore");

namespace jb = tensorstore::internal_json_binding;

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
  void MaybeSetDeadline(grpc::ClientContext& context) {
    if (spec_.timeout > absl::ZeroDuration() &&
        spec_.timeout != absl::InfiniteDuration()) {
      context.set_deadline(absl::ToChronoTime(absl::Now() + spec_.timeout));
    }
  }

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

  SpecData spec_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<KvStoreService::StubInterface> stub_;
};

////////////////////////////////////////////////////

/// Implements `TsGrpcKeyValueStore::Read`.
struct ReadTask : public internal::AtomicReferenceCount<ReadTask> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver;
  grpc::ClientContext context;
  ReadRequest request;
  ReadResponse response;

  Future<kvstore::ReadResult> Start(kvstore::Key key,
                                    const kvstore::ReadOptions& options) {
    request.set_key(std::move(key));
    request.set_generation_if_equal(
        options.generation_conditions.if_equal.value);
    request.set_generation_if_not_equal(
        options.generation_conditions.if_not_equal.value);
    if (!options.byte_range.IsFull()) {
      request.mutable_byte_range()->set_inclusive_min(
          options.byte_range.inclusive_min);
      request.mutable_byte_range()->set_exclusive_max(
          options.byte_range.exclusive_max);
    }
    if (options.staleness_bound != absl::InfiniteFuture()) {
      AbslTimeToProto(options.staleness_bound,
                      request.mutable_staleness_bound());
    }

    driver->MaybeSetDeadline(context);

    internal::IntrusivePtr<ReadTask> self(this);
    auto pair = tensorstore::PromiseFuturePair<kvstore::ReadResult>::Make();
    pair.promise.ExecuteWhenNotNeeded([self] { self->context.TryCancel(); });

    driver->stub()->async()->Read(
        &context, &request, &response,
        WithExecutor(driver->executor(), [self = std::move(self),
                                          promise = std::move(pair.promise)](
                                             ::grpc::Status s) {
          if (!promise.result_needed()) return;
          promise.SetResult(self->Ready(GrpcStatusToAbslStatus(s)));
        }));

    return std::move(pair.future);
  }

  Result<kvstore::ReadResult> Ready(absl::Status status) {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "ReadTask::Ready " << ConciseDebugString(response) << " " << status;

    TENSORSTORE_RETURN_IF_ERROR(status);
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    TENSORSTORE_ASSIGN_OR_RETURN(auto stamp,
                                 DecodeGenerationAndTimestamp(response));
    return kvstore::ReadResult{
        static_cast<kvstore::ReadResult::State>(response.state()),
        absl::Cord(response.value()),
        std::move(stamp),
    };
  }
};

/// Implements `TsGrpcKeyValueStore::Write`.
struct WriteTask : public internal::AtomicReferenceCount<WriteTask> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver;
  grpc::ClientContext context;
  WriteRequest request;
  WriteResponse response;

  Future<TimestampedStorageGeneration> Start(
      kvstore::Key key, const absl::Cord value,
      const kvstore::WriteOptions& options) {
    request.set_key(std::move(key));
    request.set_value(value);
    request.set_generation_if_equal(
        options.generation_conditions.if_equal.value);

    driver->MaybeSetDeadline(context);

    internal::IntrusivePtr<WriteTask> self(this);
    auto pair =
        tensorstore::PromiseFuturePair<TimestampedStorageGeneration>::Make();
    pair.promise.ExecuteWhenNotNeeded([self] { self->context.TryCancel(); });

    driver->stub()->async()->Write(
        &context, &request, &response,
        WithExecutor(driver->executor(), [self = std::move(self),
                                          promise = std::move(pair.promise)](
                                             ::grpc::Status s) {
          if (!promise.result_needed()) return;
          promise.SetResult(self->Ready(GrpcStatusToAbslStatus(s)));
        }));
    return std::move(pair.future);
  }

  Result<TimestampedStorageGeneration> Ready(absl::Status status) {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "WriteTask::Ready " << ConciseDebugString(response) << " " << status;
    TENSORSTORE_RETURN_IF_ERROR(status);
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    return DecodeGenerationAndTimestamp(response);
  }
};

/// Implements `TsGrpcKeyValueStore::Delete`.
struct DeleteTask : public internal::AtomicReferenceCount<DeleteTask> {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver;
  grpc::ClientContext context;
  DeleteRequest request;
  DeleteResponse response;

  Future<TimestampedStorageGeneration> Start(
      kvstore::Key key, const kvstore::WriteOptions options) {
    request.set_key(std::move(key));
    request.set_generation_if_equal(
        options.generation_conditions.if_equal.value);
    return StartImpl();
  }

  Future<TimestampedStorageGeneration> StartRange(KeyRange range) {
    request.mutable_range()->set_inclusive_min(range.inclusive_min);
    request.mutable_range()->set_exclusive_max(range.exclusive_max);
    return StartImpl();
  }

  Future<TimestampedStorageGeneration> StartImpl() {
    driver->MaybeSetDeadline(context);

    internal::IntrusivePtr<DeleteTask> self(this);
    auto pair =
        tensorstore::PromiseFuturePair<TimestampedStorageGeneration>::Make();
    pair.promise.ExecuteWhenNotNeeded([self] { self->context.TryCancel(); });

    driver->stub()->async()->Delete(
        &context, &request, &response,
        WithExecutor(driver->executor(), [self = std::move(self),
                                          promise = std::move(pair.promise)](
                                             ::grpc::Status s) {
          if (!promise.result_needed()) return;
          promise.SetResult(self->Ready(GrpcStatusToAbslStatus(s)));
        }));
    return std::move(pair.future);
  }

  Result<TimestampedStorageGeneration> Ready(absl::Status status) {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "DeleteTask::Ready " << ConciseDebugString(response) << " "
        << status;
    TENSORSTORE_RETURN_IF_ERROR(status);
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    return DecodeGenerationAndTimestamp(response);
  }
};

// Implements TsGrpcKeyValueStore::List
// NOTE: Convert to async().
struct ListTask {
  internal::IntrusivePtr<TsGrpcKeyValueStore> driver;
  ListReceiver receiver;

  grpc::ClientContext context;
  std::atomic<bool> cancelled = false;
  ListRequest request;

  ListTask(internal::IntrusivePtr<TsGrpcKeyValueStore>&& driver,
           ListReceiver&& receiver)
      : driver(std::move(driver)), receiver(std::move(receiver)) {}

  bool is_cancelled() { return cancelled.load(std::memory_order_relaxed); }

  void try_cancel() {
    if (!cancelled.load()) {
      cancelled.store(true, std::memory_order_relaxed);
      context.TryCancel();
    }
  }

  void Run() {
    // NOTE: Revisit deadline on list vs. read, etc.
    driver->MaybeSetDeadline(context);

    // Start a call.
    auto reader = driver->stub()->List(&context, request);

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

/// Key value store operations.
Future<kvstore::ReadResult> TsGrpcKeyValueStore::Read(Key key,
                                                      ReadOptions options) {
  grpc_read.Increment();
  auto task = internal::MakeIntrusivePtr<ReadTask>();
  task->driver = internal::IntrusivePtr<TsGrpcKeyValueStore>(this);
  return task->Start(std::move(key), options);
}

Future<TimestampedStorageGeneration> TsGrpcKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  if (value) {
    grpc_write.Increment();
    auto task = internal::MakeIntrusivePtr<WriteTask>();
    task->driver = internal::IntrusivePtr<TsGrpcKeyValueStore>(this);
    return task->Start(std::move(key), value.value(), options);
  } else {
    // empty value is delete.
    grpc_delete.Increment();
    auto task = internal::MakeIntrusivePtr<DeleteTask>();
    task->driver = internal::IntrusivePtr<TsGrpcKeyValueStore>(this);
    return task->Start(std::move(key), options);
  }
}

Future<const void> TsGrpcKeyValueStore::DeleteRange(KeyRange range) {
  if (range.empty()) return absl::OkStatus();
  grpc_delete_range.Increment();
  auto task = internal::MakeIntrusivePtr<DeleteTask>();
  task->driver = internal::IntrusivePtr<TsGrpcKeyValueStore>(this);

  // Convert Future<TimestampedStorageGeneration> to Future<void>
  return MapFuture(
      InlineExecutor{},
      [](const Result<TimestampedStorageGeneration>& result) {
        return MakeResult(result.status());
      },
      task->StartRange(std::move(range)));
}

void TsGrpcKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }
  grpc_list.Increment();
  auto task = std::make_unique<ListTask>(
      internal::IntrusivePtr<TsGrpcKeyValueStore>(this), std::move(receiver));
  task->request.mutable_range()->set_inclusive_min(options.range.inclusive_min);
  task->request.mutable_range()->set_exclusive_max(options.range.exclusive_max);
  task->request.set_strip_prefix_length(options.strip_prefix_length);
  if (options.staleness_bound != absl::InfiniteFuture()) {
    AbslTimeToProto(options.staleness_bound,
                    task->request.mutable_staleness_bound());
  }

  executor()([task = std::move(task)] { task->Run(); });
}

Future<kvstore::DriverPtr> TsGrpcKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<TsGrpcKeyValueStore>();
  driver->spec_ = data_;

  // Create a communication channel with credentials, then use that
  // to construct a gprc stub.
  //
  // TODO: Determine a better mapping to a grpc credentials for this.
  // grpc::Credentials ties the authentication to the communication channel
  // See: <grpcpp/security/credentials.h>, https://grpc.io/docs/guides/auth/
  ABSL_LOG_IF(INFO, verbose_logging)
      << "tsgrpc_kvstore address=" << data_.address;

  driver->channel_ =
      grpc::CreateChannel(data_.address, data_.credentials->GetCredentials());

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
