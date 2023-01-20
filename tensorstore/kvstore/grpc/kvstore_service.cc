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

#include "tensorstore/kvstore/grpc/kvstore_service.h"

#include <stddef.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include <nlohmann/json.hpp>
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/grpc/common.h"
#include "tensorstore/kvstore/grpc/common.pb.h"
#include "tensorstore/kvstore/grpc/handler_template.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

// grpc/proto

#include "google/protobuf/timestamp.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.pb.h"

using ::grpc::CallbackServerContext;
using ::tensorstore::PromiseFuturePair;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::ProtoToAbslTime;
using ::tensorstore::kvstore::Key;
using ::tensorstore::kvstore::ReadResult;
using ::tensorstore_grpc::EncodeGenerationAndTimestamp;
using ::tensorstore_grpc::Handler;
using ::tensorstore_grpc::StreamHandler;
using ::tensorstore_grpc::kvstore::DeleteRequest;
using ::tensorstore_grpc::kvstore::DeleteResponse;
using ::tensorstore_grpc::kvstore::ListRequest;
using ::tensorstore_grpc::kvstore::ListResponse;
using ::tensorstore_grpc::kvstore::ReadRequest;
using ::tensorstore_grpc::kvstore::ReadResponse;
using ::tensorstore_grpc::kvstore::WriteRequest;
using ::tensorstore_grpc::kvstore::WriteResponse;
using ::tensorstore_grpc::kvstore::grpc_gen::KvStoreService;

namespace {
[[maybe_unused]] inline void MyCopyTo(const absl::Cord& src, std::string* dst) {
  absl::CopyCordToString(src, dst);
}
[[maybe_unused]] inline void MyCopyTo(const absl::Cord& src, absl::Cord* dst) {
  *dst = src;
}

class ReadHandler final : public Handler<ReadRequest, ReadResponse> {
  using Base = Handler<ReadRequest, ReadResponse>;

 public:
  ReadHandler(CallbackServerContext* grpc_context, const Request* request,
              Response* response, tensorstore::KvStore kvstore)
      : Base(grpc_context, request, response), kvstore_(std::move(kvstore)) {}

  void Run() {
    tensorstore::kvstore::ReadOptions options{};
    options.if_equal.value = request()->generation_if_equal();
    options.if_not_equal.value = request()->generation_if_not_equal();

    if (request()->has_byte_range()) {
      options.byte_range.inclusive_min =
          request()->byte_range().inclusive_min();
      if (request()->byte_range().exclusive_max() != 0) {
        options.byte_range.exclusive_max =
            request()->byte_range().exclusive_max();
      }
    }
    if (request()->has_staleness_bound()) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          options.staleness_bound,
          ProtoToAbslTime(request()->staleness_bound()), Finish(_));
    }

    IntrusivePtr<ReadHandler> self{this};
    future_ =
        PromiseFuturePair<void>::Link(
            [self = std::move(self)](tensorstore::Promise<void> promise,
                                     auto read_result) {
              if (!promise.result_needed()) return;
              promise.SetResult(self->HandleResult(read_result.result()));
            },
            tensorstore::kvstore::Read(kvstore_, request()->key(), options))
            .future;
  }

  void OnCancel() final {
    if (future_.ready()) return;
    future_ = {};
    Finish(::grpc::Status(::grpc::StatusCode::CANCELLED, ""));
  }

  absl::Status HandleResult(const tensorstore::Result<ReadResult>& result) {
    auto status = result.status();
    if (status.ok()) {
      auto& r = result.value();
      response()->set_state(static_cast<ReadResponse::State>(r.state));
      EncodeGenerationAndTimestamp(r.stamp, response());
      if (r.has_value()) {
        MyCopyTo(r.value, response()->mutable_value());
      }
    }
    Finish(status);
    return status;
  }

 private:
  tensorstore::KvStore kvstore_;
  tensorstore::Future<void> future_;
};

class WriteHandler final : public Handler<WriteRequest, WriteResponse> {
  using Base = Handler<WriteRequest, WriteResponse>;

 public:
  WriteHandler(CallbackServerContext* grpc_context, const Request* request,
               Response* response, tensorstore::KvStore kvstore)
      : Base(grpc_context, request, response), kvstore_(std::move(kvstore)) {}

  void Run() {
    tensorstore::kvstore::WriteOptions options{};
    options.if_equal.value = request()->generation_if_equal();

    IntrusivePtr<WriteHandler> self{this};
    future_ =
        PromiseFuturePair<void>::Link(
            [self = std::move(self)](tensorstore::Promise<void> promise,
                                     auto write_result) {
              if (!promise.result_needed()) return;
              promise.SetResult(self->HandleResult(write_result.result()));
            },
            tensorstore::kvstore::Write(kvstore_, request()->key(),
                                        absl::Cord(request()->value()),
                                        options))
            .future;
  }

  void OnCancel() final {
    if (future_.ready()) return;
    future_ = {};
    Finish(::grpc::Status(::grpc::StatusCode::CANCELLED, ""));
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

class DeleteHandler final : public Handler<DeleteRequest, DeleteResponse> {
  using Base = Handler<DeleteRequest, DeleteResponse>;

 public:
  DeleteHandler(CallbackServerContext* grpc_context, const Request* request,
                Response* response, tensorstore::KvStore kvstore)
      : Base(grpc_context, request, response), kvstore_(std::move(kvstore)) {}

  void Run() {
    IntrusivePtr<DeleteHandler> self{this};
    auto callback = [self = std::move(self)](tensorstore::Promise<void> promise,
                                             auto del_result) {
      if (!promise.result_needed()) return;
      promise.SetResult(self->HandleResult(del_result.result()));
    };

    if (request()->has_range()) {
      future_ = PromiseFuturePair<void>::Link(
                    std::move(callback),
                    tensorstore::kvstore::DeleteRange(
                        kvstore_, tensorstore::KeyRange(
                                      request()->range().inclusive_min(),
                                      request()->range().exclusive_max())))
                    .future;
    } else if (!request()->key().empty()) {
      tensorstore::kvstore::WriteOptions options{};
      options.if_equal.value = request()->generation_if_equal();

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

class ListHandler final : public StreamHandler<ListRequest, ListResponse> {
  using Base = StreamHandler<ListRequest, ListResponse>;

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
      } else if (current_->key().empty()) {
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
      IntrusivePtr<ListHandler>& self, tensorstore::AnyCancelReceiver cancel) {
    absl::MutexLock l(&self->mu_);
    self->cancel_ = std::move(cancel);
    self->done_ = false;
    self->current_ = std::make_unique<ListResponse>();
    self->estimated_size_ = 0;
  }

  [[maybe_unused]] friend void set_value(IntrusivePtr<ListHandler>& self,
                                         Key key) {
    absl::MutexLock l(&self->mu_);
    self->current_->add_key(key);
    self->estimated_size_ += key.size();
    self->MaybeWrite();
  }

  [[maybe_unused]] friend void set_done(IntrusivePtr<ListHandler>& self) {
    self->cancel_ = [] {};
  }

  [[maybe_unused]] friend void set_error(IntrusivePtr<ListHandler>& self,
                                         absl::Status s) {
    absl::MutexLock l(&self->mu_);
    self->cancel_ = [] {};
    self->status_ = s;
  }

  [[maybe_unused]] friend void set_stopping(IntrusivePtr<ListHandler>& self) {
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
  tensorstore::kvstore::ListOptions options;
  options.range = tensorstore::KeyRange(request()->range().inclusive_min(),
                                        request()->range().exclusive_max());
  options.strip_prefix_length = request()->strip_prefix_length();

  IntrusivePtr<ListHandler> self{this};
  tensorstore::execution::submit(
      tensorstore::kvstore::List(self->kvstore_, options), self);
}

// ---------------------------------------

}  // namespace
namespace tensorstore_grpc {

KvStoreServiceImpl::KvStoreServiceImpl(tensorstore::KvStore kvstore)
    : kvstore_(std::move(kvstore)) {}

TENSORSTORE_GRPC_HANDLER(KvStoreServiceImpl::Read, ReadHandler, kvstore_);
TENSORSTORE_GRPC_HANDLER(KvStoreServiceImpl::Write, WriteHandler, kvstore_);
TENSORSTORE_GRPC_HANDLER(KvStoreServiceImpl::Delete, DeleteHandler, kvstore_);
TENSORSTORE_GRPC_STREAM_HANDLER(KvStoreServiceImpl::List, ListHandler,
                                kvstore_);

}  // namespace tensorstore_grpc
