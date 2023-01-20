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

#include "tensorstore/kvstore/grpc/grpc_kvstore.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>

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
#include "grpcpp/support/time.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/concurrency_resource_provider.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/grpc/common.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/serialization/absl_time.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// protos

#include "google/protobuf/timestamp.pb.h"
#include "tensorstore/kvstore/grpc/common.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.pb.h"

using ::tensorstore::GrpcClientCredentials;
using ::tensorstore::internal::AbslTimeToProto;
using ::tensorstore::internal::GrpcStatusToAbslStatus;
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

[[maybe_unused]] inline void MyCopyTo(const absl::Cord& src, std::string* dst) {
  absl::CopyCordToString(src, dst);
}

[[maybe_unused]] inline void MyCopyTo(const absl::Cord& src, absl::Cord* dst) {
  *dst = src;
}

//////////////////////////
#define TENSORSTORE_INTERNAL_TASK_IMPL(Op, PromiseValue)                      \
  static Future<PromiseValue> Start(KvStoreService::StubInterface* stub,      \
                                    std::shared_ptr<Op##Task> task,           \
                                    absl::Time deadline) {                    \
    auto pair = tensorstore::PromiseFuturePair<PromiseValue>::Make();         \
    task->promise = std::move(pair.promise);                                  \
    task->context.set_deadline(absl::ToChronoTime(deadline));                 \
    task->registration = task->promise.ExecuteWhenNotNeeded(                  \
        [task] { task->context.TryCancel(); });                               \
    stub->async()->Op(&task->context, &task->request, &task->response,        \
                      [task](::grpc::Status s) {                              \
                        if (!task->promise.result_needed()) return;           \
                        task->registration.Unregister();                      \
                        if (!s.ok()) {                                        \
                          task->promise.SetResult(GrpcStatusToAbslStatus(s)); \
                        } else {                                              \
                          task->promise.SetResult(task->Ready());             \
                        }                                                     \
                      });                                                     \
    return pair.future;                                                       \
  }                                                                           \
  grpc::ClientContext context;                                                \
  tensorstore::Promise<PromiseValue> promise;                                 \
  FutureCallbackRegistration registration

/// Implements `GrpcKeyValueStore::Read`.
struct ReadTask {
  TENSORSTORE_INTERNAL_TASK_IMPL(Read, kvstore::ReadResult);

  ReadRequest request;
  ReadResponse response;

  Result<kvstore::ReadResult> Ready() {
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    kvstore::ReadResult result;
    TENSORSTORE_ASSIGN_OR_RETURN(result.stamp,
                                 DecodeGenerationAndTimestamp(response));
    result.state = static_cast<kvstore::ReadResult::State>(response.state());
    if (result.has_value()) {
      result.value = response.value();
    }
    return result;
  }
};

/// Implements `GrpcKeyValueStore::Write`.
struct WriteTask {
  TENSORSTORE_INTERNAL_TASK_IMPL(Write, TimestampedStorageGeneration);

  WriteRequest request;
  WriteResponse response;

  Result<TimestampedStorageGeneration> Ready() {
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    return DecodeGenerationAndTimestamp(response);
  }
};

/// Implements `GrpcKeyValueStore::Delete`.
struct DeleteTask {
  TENSORSTORE_INTERNAL_TASK_IMPL(Delete, TimestampedStorageGeneration);

  DeleteRequest request;
  DeleteResponse response;

  Result<TimestampedStorageGeneration> Ready() {
    TENSORSTORE_RETURN_IF_ERROR(GetMessageStatus(response));
    return DecodeGenerationAndTimestamp(response);
  }
};

#undef TENSORSTORE_INTERNAL_TASK_IMPL

// Implements GrpcKeyValueStore::List
struct ListTask {
  grpc::ClientContext context;
  std::atomic<bool> cancelled = false;
  AnyFlowReceiver<absl::Status, kvstore::Key> receiver;
  ListRequest request;

  bool is_cancelled() { return cancelled.load(std::memory_order_relaxed); }

  void try_cancel() {
    if (!cancelled.load()) {
      cancelled.store(true, std::memory_order_relaxed);
      context.TryCancel();
    }
  }

  void Run(KvStoreService::StubInterface* stub, absl::Time deadline) {
    context.set_deadline(absl::ToChronoTime(deadline));
    // Start a call.
    auto reader = stub->List(&context, request);

    execution::set_starting(receiver, [this] { try_cancel(); });

    absl::Status msg_status;
    ListResponse response;
    while (reader->Read(&response)) {
      msg_status = GetMessageStatus(response);
      if (!msg_status.ok()) {
        try_cancel();
        break;
      }
      for (const auto& k : response.key()) {
        execution::set_value(receiver, k);
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

struct GrpcKeyValueStoreSpecData {
  std::string address;
  absl::Duration timeout;
  Context::Resource<GrpcClientCredentials> credentials;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.address, x.timeout, x.credentials);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member(GrpcClientCredentials::id,
                 jb::Projection<&GrpcKeyValueStoreSpecData::credentials>()),
      jb::Member("address",
                 jb::Projection<&GrpcKeyValueStoreSpecData::address>()),
      jb::Member("timeout", jb::Projection<&GrpcKeyValueStoreSpecData::timeout>(
                                jb::DefaultValue([](auto* x) {
                                  *x = absl::Minutes(5);
                                }))) /**/
  );
};

class GrpcKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<GrpcKeyValueStoreSpec,
                                                    GrpcKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "grpc_kvstore";
  Future<kvstore::DriverPtr> DoOpen() const override;
};

/// Defines the "grpc_kvstore" KeyValueStore driver.
class GrpcKeyValueStore
    : public internal_kvstore::RegisteredDriver<GrpcKeyValueStore,
                                                GrpcKeyValueStoreSpec> {
 public:
  absl::Time GetTimeout() {
    if (spec_.timeout == absl::InfiniteDuration()) {
      return absl::Now() + absl::Minutes(10);
    }
    return absl::Now() + spec_.timeout;
  }

  /// Key value store operations.
  Future<ReadResult> Read(Key key, ReadOptions options) override {
    auto task = std::make_shared<ReadTask>();
    task->request.set_key(std::move(key));
    task->request.set_generation_if_equal(options.if_equal.value);
    task->request.set_generation_if_not_equal(options.if_not_equal.value);
    if (options.byte_range.inclusive_min != 0 ||
        options.byte_range.exclusive_max != std::nullopt) {
      task->request.mutable_byte_range()->set_inclusive_min(
          options.byte_range.inclusive_min);
      if (options.byte_range.exclusive_max != std::nullopt) {
        task->request.mutable_byte_range()->set_exclusive_max(
            *options.byte_range.exclusive_max);
      }
    }
    if (options.staleness_bound != absl::InfiniteFuture()) {
      AbslTimeToProto(options.staleness_bound,
                      task->request.mutable_staleness_bound());
    }
    return ReadTask::Start(stub_.get(), std::move(task), GetTimeout());
  }

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override {
    if (value) {
      auto task = std::make_shared<WriteTask>();
      task->request.set_key(std::move(key));
      MyCopyTo(*value, task->request.mutable_value());
      task->request.set_generation_if_equal(options.if_equal.value);

      return WriteTask::Start(stub_.get(), std::move(task), GetTimeout());
    } else {
      // empty value is delete.
      auto task = std::make_shared<DeleteTask>();
      task->request.set_key(std::move(key));
      task->request.set_generation_if_equal(options.if_equal.value);

      return DeleteTask::Start(stub_.get(), std::move(task), GetTimeout());
    }
  }

  Future<const void> DeleteRange(KeyRange range) override {
    if (range.empty()) return absl::OkStatus();
    auto task = std::make_shared<DeleteTask>();
    task->request.mutable_range()->set_inclusive_min(range.inclusive_min);
    task->request.mutable_range()->set_exclusive_max(range.exclusive_max);

    // Convert Future<TimestampedStorageGeneration> to Future<void>
    return MapFuture(
        InlineExecutor{},
        [](const Result<TimestampedStorageGeneration>& result) {
          return MakeResult(result.status());
        },
        DeleteTask::Start(stub_.get(), std::move(task), GetTimeout()));
  }

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override {
    if (options.range.empty()) {
      execution::set_starting(receiver, [] {});
      execution::set_done(receiver);
      execution::set_stopping(receiver);
      return;
    }
    auto task = std::make_unique<ListTask>();
    task->receiver = std::move(receiver);
    task->request.mutable_range()->set_inclusive_min(
        options.range.inclusive_min);
    task->request.mutable_range()->set_exclusive_max(
        options.range.exclusive_max);

    task->Run(stub_.get(), GetTimeout());
  }

  /// Obtains a `SpecData` representation from an open `Driver`.
  absl::Status GetBoundSpecData(SpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  absl::Status OpenImpl() {
    // TODO: Determine a better mapping to a grpc credentials for this.
    // grpc::Credentials ties the authentication to the communication channel
    // See: <grpcpp/security/credentials.h>, https://grpc.io/docs/guides/auth/
    if (!stub_) {
      // Create a communication channel with credentials, then use that
      // to construct a gprc stub.
      ABSL_LOG(INFO) << "grpc_kvstore address=" << spec_.address;
      channel_ = grpc::CreateChannel(spec_.address,
                                     spec_.credentials->GetCredentials());
      stub_ = KvStoreService::NewStub(channel_);
    }

    return absl::OkStatus();
  }

  SpecData spec_;
  std::shared_ptr<KvStoreService::StubInterface> stub_;
  std::shared_ptr<grpc::Channel> channel_;
};

Future<kvstore::DriverPtr> GrpcKeyValueStoreSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<GrpcKeyValueStore>();
  driver->spec_ = data_;
  TENSORSTORE_RETURN_IF_ERROR(driver->OpenImpl());
  return driver;
}

}  // namespace

Result<kvstore::DriverPtr> CreateGrpcKvStore(
    std::string address, absl::Duration timeout,
    std::shared_ptr<KvStoreService::StubInterface> stub) {
  auto ptr = internal::MakeIntrusivePtr<GrpcKeyValueStore>();
  ptr->spec_.address = std::move(address);
  ptr->spec_.timeout = timeout;
  if (stub) {
    ptr->stub_ = std::move(stub);
  }
  TENSORSTORE_RETURN_IF_ERROR(ptr->OpenImpl());
  return ptr;
}

}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::GrpcKeyValueStore)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::GrpcKeyValueStoreSpec>
    registration;
}
