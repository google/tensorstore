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

#include "tensorstore/kvstore/tsgrpc/kvstore_server.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/grpc/server_credentials.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/tsgrpc/common.h"
#include "tensorstore/kvstore/tsgrpc/common.pb.h"
#include "tensorstore/kvstore/tsgrpc/handler_template.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

// grpc/proto
#include "tensorstore/kvstore/tsgrpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/tsgrpc/kvstore.pb.h"
#include "tensorstore/util/span.h"

using ::grpc::CallbackServerContext;
using ::tensorstore::internal_metrics::MetricMetadata;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore_grpc::EncodeGenerationAndTimestamp;
using ::tensorstore_grpc::Handler;
using ::tensorstore_grpc::StreamClientRequestHandler;
using ::tensorstore_grpc::StreamServerResponseHandler;
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

namespace jb = ::tensorstore::internal_json_binding;

auto& read_metric = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc_server/read",
    MetricMetadata("KvStoreService::Read calls"));

auto& write_metric = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc_server/write",
    MetricMetadata("KvStoreService::Write calls"));

auto& delete_metric = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc_server/delete",
    MetricMetadata("KvStoreService::Delete calls"));

auto& list_metric = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/tsgrpc_server/list",
    MetricMetadata("KvStoreService::List calls"));

ABSL_CONST_INIT internal_log::VerboseFlag verbose_logging("tsgrpc_kvstore");

constexpr size_t kMaxReadChunkSize = 1 << 20;

class ReadHandler final
    : public StreamServerResponseHandler<ReadRequest, ReadResponse> {
  using Base = StreamServerResponseHandler<ReadRequest, ReadResponse>;

 public:
  ReadHandler(CallbackServerContext* grpc_context, const Request* request,
              KvStore kvstore)
      : Base(grpc_context, request), kvstore_(std::move(kvstore)) {}

  void Run() {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "ReadHandler " << ConciseDebugString(*request());
    kvstore::ReadOptions options{};
    options.generation_conditions.if_equal.value =
        request()->generation_if_equal();
    options.generation_conditions.if_not_equal.value =
        request()->generation_if_not_equal();

    if (request()->has_byte_range()) {
      options.byte_range.inclusive_min =
          request()->byte_range().inclusive_min();
      options.byte_range.exclusive_max =
          request()->byte_range().exclusive_max();
      if (!options.byte_range.SatisfiesInvariants()) {
        Finish(absl::InvalidArgumentError("Invalid byte range"));
        return;
      }
    }
    if (request()->has_staleness_bound()) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          options.staleness_bound,
          internal::ProtoToAbslTime(request()->staleness_bound()), Finish(_));
    }

    internal::IntrusivePtr<ReadHandler> self{this};
    future_ = tensorstore::kvstore::Read(kvstore_, request()->key(), options);
    future_.ExecuteWhenReady(
        [self = std::move(self)](ReadyFuture<kvstore::ReadResult> ready) {
          self->HandleInitialResult(std::move(ready).result());
        });
  }

  void HandleInitialResult(Result<kvstore::ReadResult> result) {
    auto status = result.status();
    if (!status.ok()) {
      // Consider setting the status in the response instead.
      Finish(status);
      return;
    }

    auto& r = result.value();
    response_.set_state(static_cast<ReadResponse::State>(r.state));
    EncodeGenerationAndTimestamp(r.stamp, &response_);

    value_ = std::move(r.value);
    value_offset_ = 0;

    SetNextPart();
    StartWrite(&response_);
  }

  void SetNextPart() {
    auto next_part = value_.Subcord(value_offset_, kMaxReadChunkSize);
    value_offset_ = std::min(value_.size(), value_offset_ + next_part.size());
    response_.set_value_part(std::move(next_part));
  }

  void OnCancel() final {
    if (future_.ready()) return;
    future_ = {};
    Finish(::grpc::Status(::grpc::StatusCode::CANCELLED, ""));
  }

  void OnWriteDone(bool ok) final {
    if (!ok) {
      // OnDone is going to be called after we return from this method.
      Finish(::grpc::Status(::grpc::StatusCode::UNKNOWN, "Write failed"));
      return;
    }
    if (value_offset_ == value_.size()) {
      Finish(::grpc::Status::OK);
      return;
    }
    response_.Clear();
    SetNextPart();
    StartWrite(&response_);
  }

 private:
  KvStore kvstore_;
  Future<kvstore::ReadResult> future_;

  ReadResponse response_;
  absl::Cord value_;
  size_t value_offset_ = 0;
};

class WriteHandler final
    : public StreamClientRequestHandler<WriteRequest, WriteResponse> {
  using Base = StreamClientRequestHandler<WriteRequest, WriteResponse>;

 public:
  WriteHandler(CallbackServerContext* grpc_context, Response* response,
               KvStore kvstore)
      : Base(grpc_context, response), kvstore_(std::move(kvstore)) {}

  void Run() { StartRead(&request_); }

  void OnReadDone(bool ok) override {
    if (grpc_context()->IsCancelled()) {
      // OnCancelled is going to be called and will invoke Finish, thus just
      // stop reading sequence here.
      return;
    }
    if (ok) {
      ABSL_LOG_IF(INFO, verbose_logging)
          << "WriteHandler " << ConciseDebugString(request_);
      // One read chunk has completed, but not everything.
      // According to protocol, state, generation and timestamp must be taken
      // from the first message in the stream.
      if (!first_request_received_) {
        first_request_received_ = true;
        options_.generation_conditions.if_equal.value =
            request_.generation_if_equal();
        key_ = request_.key();
      }

      value_.Append(request_.value_part());
      StartRead(&request_);
      return;
    }

    ABSL_LOG_IF(INFO, verbose_logging) << "WriteHandler starting write";

    // Setup the response.
    internal::IntrusivePtr<WriteHandler> self{this};
    future_ = tensorstore::kvstore::Write(kvstore_, key_, value_, options_);
    future_.ExecuteWhenReady(
        [self =
             std::move(self)](ReadyFuture<TimestampedStorageGeneration> ready) {
          self->HandleResult(std::move(ready).result());
        });
  }

  void OnCancel() final {
    if (future_.ready()) return;
    future_ = {};
    Finish(::grpc::Status(::grpc::StatusCode::CANCELLED, ""));
  }

  void HandleResult(tensorstore::Result<TimestampedStorageGeneration>& result) {
    auto status = result.status();
    if (status.ok()) {
      EncodeGenerationAndTimestamp(result.value(), response());
    }
    Finish(status);
  }

 private:
  KvStore kvstore_;
  tensorstore::kvstore::WriteOptions options_;

  Future<TimestampedStorageGeneration> future_;
  WriteRequest request_;

  std::string key_;
  absl::Cord value_;
  bool first_request_received_ = false;
};

class DeleteHandler final : public Handler<DeleteRequest, DeleteResponse> {
  using Base = Handler<DeleteRequest, DeleteResponse>;

 public:
  DeleteHandler(CallbackServerContext* grpc_context, const Request* request,
                Response* response, KvStore kvstore)
      : Base(grpc_context, request, response), kvstore_(std::move(kvstore)) {}

  void Run() {
    ABSL_LOG_IF(INFO, verbose_logging)
        << "DeleteHandler " << ConciseDebugString(*request());
    internal::IntrusivePtr<DeleteHandler> self{this};
    auto callback = [self = std::move(self)](Promise<void> promise,
                                             auto del_result) {
      if (!promise.result_needed()) return;
      promise.SetResult(self->HandleResult(del_result.result()));
    };

    if (request()->has_range()) {
      future_ = PromiseFuturePair<void>::Link(
                    std::move(callback),
                    kvstore::DeleteRange(
                        kvstore_, KeyRange(request()->range().inclusive_min(),
                                           request()->range().exclusive_max())))
                    .future;
    } else if (!request()->key().empty()) {
      kvstore::WriteOptions options{};
      options.generation_conditions.if_equal.value =
          request()->generation_if_equal();

      future_ =
          PromiseFuturePair<void>::Link(
              std::move(callback),
              tensorstore::kvstore::Delete(kvstore_, request()->key(), options))
              .future;

    } else {
      Finish(absl::InvalidArgumentError("Invalid request"));
    }
  }

  void OnCancel() final {
    if (future_.ready()) return;
    future_ = {};
    Finish(::grpc::Status(::grpc::StatusCode::CANCELLED, ""));
  }

  absl::Status HandleResult(const tensorstore::Result<void>& result) {
    auto status = result.status();
    Finish(status);
    return status;
  }

  absl::Status HandleResult(
      const tensorstore::Result<TimestampedStorageGeneration>& result) {
    auto status = result.status();
    if (status.ok()) {
      EncodeGenerationAndTimestamp(result.value(), response());
    }
    Finish(status);
    return status;
  }

 private:
  tensorstore::KvStore kvstore_;
  tensorstore::Future<void> future_;
};

class ListHandler final
    : public StreamServerResponseHandler<ListRequest, ListResponse> {
  using Base = StreamServerResponseHandler<ListRequest, ListResponse>;

 public:
  ListHandler(CallbackServerContext* grpc_context, const Request* request,
              tensorstore::KvStore kvstore)
      : Base(grpc_context, request),
        kvstore_(std::move(kvstore)),
        estimated_size_(0),
        cancel_([] {}) {}

  void Run();

  void OnCancel() final { cancel_(); }

  void OnWriteDone(bool ok) final {
    absl::MutexLock l(&mu_);
    in_flight_msg_ = nullptr;
    MaybeWrite();
  }

  /// MaybeWrite potentially enqueues the next required write.
  /// When no write is possible (an existing message is in flight, or
  /// the pending message is "empty"), then it does nothing.
  /// Otherwise it starts sending the the pending message and allocates
  /// a new buffer.
  void MaybeWrite() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    // A message is still in-flight; only 1 allowed at a time.
    if (in_flight_msg_ != nullptr) return;

    // No pending message / final message already sent.
    if (!current_) return;

    // The sender has completed, so apply final processing.
    if (done_) {
      if (!status_.ok()) {
        // List failed, return status.
        current_ = nullptr;
        Finish(status_);
      } else if (current_->entry().empty()) {
        // List succeeded, however the message is empty.
        current_ = nullptr;
        Finish(grpc::Status::OK);
      } else {
        // Send last pending proto.
        // StartWriteAndFinish does call OnWriteDone, only OnDone.
        in_flight_msg_ = std::move(current_);
        StartWriteAndFinish(in_flight_msg_.get(), {}, grpc::Status::OK);
      }
      return;
    }

    // NOTE: There's no mechanism for the reactor to send multiple messages
    // and indicate which message was sent, so we track a single
    // in_flight_message_.
    //
    // There's also very little control flow available on the sender/receiver
    // mechanism, which could be useful here.

    // Nothing to send. This selection/rejection criteria should be adapted
    // based on what works well.

    constexpr size_t kTargetSize = 16 * 1024;  // look for a minimum 16kb proto.
    if (estimated_size_ < kTargetSize) return;

    in_flight_msg_ = std::move(current_);
    StartWrite(in_flight_msg_.get());

    current_ = std::make_unique<ListResponse>();
    estimated_size_ = 0;
  }

  /// AnyFlowReceiver methods.
  [[maybe_unused]] friend void set_starting(
      internal::IntrusivePtr<ListHandler>& self, AnyCancelReceiver cancel) {
    absl::MutexLock l(&self->mu_);
    self->cancel_ = std::move(cancel);
    self->done_ = false;
    self->current_ = std::make_unique<ListResponse>();
    self->estimated_size_ = 0;
  }

  [[maybe_unused]] friend void set_value(
      internal::IntrusivePtr<ListHandler>& self, ListEntry entry) {
    absl::MutexLock l(&self->mu_);
    auto* e = self->current_->add_entry();
    e->set_key(entry.key);
    e->set_size(entry.size);
    self->estimated_size_ += entry.key.size();
    self->MaybeWrite();
  }

  [[maybe_unused]] friend void set_done(
      internal::IntrusivePtr<ListHandler>& self) {
    self->cancel_ = [] {};
  }

  [[maybe_unused]] friend void set_error(
      internal::IntrusivePtr<ListHandler>& self, absl::Status s) {
    absl::MutexLock l(&self->mu_);
    self->cancel_ = [] {};
    self->status_ = s;
  }

  [[maybe_unused]] friend void set_stopping(
      internal::IntrusivePtr<ListHandler>& self) {
    absl::MutexLock l(&self->mu_);
    self->done_ = true;
    self->MaybeWrite();
  }

 private:
  tensorstore::KvStore kvstore_;

  absl::Mutex mu_;
  absl::Status status_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<ListResponse> current_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<ListResponse> in_flight_msg_ ABSL_GUARDED_BY(mu_);
  size_t estimated_size_ ABSL_GUARDED_BY(mu_);
  tensorstore::AnyCancelReceiver cancel_;
  std::atomic<bool> done_{true};
};

void ListHandler::Run() {
  ABSL_LOG_IF(INFO, verbose_logging)
      << "ListHandler " << ConciseDebugString(*request());

  tensorstore::kvstore::ListOptions options;
  options.range = tensorstore::KeyRange(request()->range().inclusive_min(),
                                        request()->range().exclusive_max());
  options.strip_prefix_length = request()->strip_prefix_length();
  if (request()->has_staleness_bound()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        options.staleness_bound,
        internal::ProtoToAbslTime(request()->staleness_bound()), Finish(_));
  }

  internal::IntrusivePtr<ListHandler> self{this};
  tensorstore::execution::submit(
      tensorstore::kvstore::List(self->kvstore_, options), self);
}

// ---------------------------------------

}  // namespace
namespace grpc_kvstore {

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    KvStoreServer::Spec,
    jb::Object(jb::Member("base", jb::Projection<&KvStoreServer::Spec::base>()),
               jb::Initialize([](auto* obj) {
                 internal::EnsureDirectoryPath(obj->base.path);
                 return absl::OkStatus();
               }),
               jb::Member("bind_addresses",
                          jb::Projection<&KvStoreServer::Spec::bind_addresses>(
                              jb::DefaultInitializedValue()))));

/// Default forwarding implementation of tensorstore_grpc::KvStoreService.
class KvStoreServer::Impl final : public KvStoreService::CallbackService {
 public:
  Impl(KvStore kvstore) : kvstore_(std::move(kvstore)) {}

  ::grpc::ServerWriteReactor<::tensorstore_grpc::kvstore::ReadResponse>* Read(
      ::grpc::CallbackServerContext* context,
      const ReadRequest* request) override {
    read_metric.Increment();
    internal::IntrusivePtr<ReadHandler> handler(
        new ReadHandler(context, request, kvstore_));
    assert(handler->use_count() == 2);
    handler->Run();
    assert(handler->use_count() > 0);
    if (handler->use_count() == 1) return nullptr;
    return handler.get();
  }

  ::grpc::ServerReadReactor<::tensorstore_grpc::kvstore::WriteRequest>* Write(
      ::grpc::CallbackServerContext* context,
      WriteResponse* response) override {
    write_metric.Increment();
    internal::IntrusivePtr<WriteHandler> handler(
        new WriteHandler(context, response, kvstore_));
    assert(handler->use_count() == 2);
    handler->Run();
    assert(handler->use_count() > 0);
    if (handler->use_count() == 1) return nullptr;
    return handler.get();
  }

  ::grpc::ServerUnaryReactor* Delete(::grpc::CallbackServerContext* context,
                                     const DeleteRequest* request,
                                     DeleteResponse* response) override {
    delete_metric.Increment();
    internal::IntrusivePtr<DeleteHandler> handler(
        new DeleteHandler(context, request, response, kvstore_));
    assert(handler->use_count() == 2);
    handler->Run();
    assert(handler->use_count() > 0);
    if (handler->use_count() == 1) return nullptr;
    return handler.get();
  }

  ::grpc::ServerWriteReactor<::tensorstore_grpc::kvstore::ListResponse>* List(
      ::grpc::CallbackServerContext* context,
      const ListRequest* request) override {
    list_metric.Increment();
    internal::IntrusivePtr<ListHandler> handler(
        new ListHandler(context, request, kvstore_));
    assert(handler->use_count() == 2);
    handler->Run();
    if (handler->use_count() == 1) return nullptr;
    return handler.get();
  }

  // Accessor
  const KvStore& kvstore() const { return kvstore_; }

 private:
  friend class KvStoreServer;
  KvStore kvstore_;
  std::vector<int> listening_ports_;
  std::unique_ptr<grpc::Server> server_;
};

KvStoreServer::KvStoreServer() = default;
KvStoreServer::~KvStoreServer() = default;
KvStoreServer::KvStoreServer(KvStoreServer&&) = default;
KvStoreServer& KvStoreServer::operator=(KvStoreServer&&) = default;

tensorstore::span<const int> KvStoreServer::ports() const {
  return impl_->listening_ports_;
}

int KvStoreServer::port() const { return impl_->listening_ports_.front(); }

/// Waits for the server to shutdown.
void KvStoreServer::Wait() { impl_->server_->Wait(); }

tensorstore::Result<KvStoreServer> KvStoreServer::Start(Spec spec,
                                                        Context context) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto kv, tensorstore::kvstore::Open(spec.base, context).result());

  auto impl = std::make_unique<KvStoreServer::Impl>(std::move(kv));

  /// FIXME: Use a bound spec for credentials.
  auto strategy = context.GetResource<tensorstore::GrpcServerCredentials>()
                      .value()
                      ->GetAuthenticationStrategy();

  grpc::ServerBuilder builder;
  builder.RegisterService(impl.get());
  strategy->AddBuilderParameters(builder);
  if (spec.bind_addresses.empty()) {
    spec.bind_addresses.push_back("[::]:0");
  }
  impl->listening_ports_.resize(spec.bind_addresses.size());
  auto creds = strategy->GetServerCredentials();
  for (size_t i = 0; i < spec.bind_addresses.size(); ++i) {
    builder.AddListeningPort(spec.bind_addresses[i], creds,
                             &impl->listening_ports_[i]);
  }
  impl->server_ = builder.BuildAndStart();

  KvStoreServer server;
  server.impl_ = std::move(impl);
  return server;
}

}  // namespace grpc_kvstore
}  // namespace tensorstore
