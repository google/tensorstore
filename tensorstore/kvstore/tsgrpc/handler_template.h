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

#ifndef TENSORSTORE_KVSTORE_TSGRPC_HANDLER_TEMPLATE_H_
#define TENSORSTORE_KVSTORE_TSGRPC_HANDLER_TEMPLATE_H_

#include <cassert>

#include "absl/status/status.h"
#include "grpcpp/grpcpp.h"  // third_party
#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore_grpc {

class HandlerBase
    : public tensorstore::internal::AtomicReferenceCount<HandlerBase> {
 public:
  using Ptr = tensorstore::internal::IntrusivePtr<HandlerBase>;

  HandlerBase(::grpc::CallbackServerContext* grpc_context)
      : grpc_context_(grpc_context) {
    // This refcount should be adopted by calling Adopt.
    intrusive_ptr_increment(this);
  }

  virtual ~HandlerBase() = default;

  // Adopt the initial refcount.
  Ptr Adopt() { return Ptr{this, tensorstore::internal::adopt_object_ref}; }

  ::grpc::CallbackServerContext* grpc_context() { return grpc_context_; }

 private:
  ::grpc::CallbackServerContext* grpc_context_;
};

template <typename RequestProto, typename ResponseProto>
class Handler : public HandlerBase, public grpc::ServerUnaryReactor {
 public:
  using Request = RequestProto;
  using Response = ResponseProto;
  using Reactor = grpc::ServerUnaryReactor;

  Handler(::grpc::CallbackServerContext* grpc_context, const Request* request,
          Response* response)
      : HandlerBase(grpc_context), request_(request), response_(response) {}

  using Reactor::Finish;
  void Finish(absl::Status status) {
    Finish(tensorstore::internal::AbslStatusToGrpcStatus(status));
  }

  const Request* request() { return request_; }
  Response* response() { return response_; }

 protected:
  void OnDone() final { auto adopted = Adopt(); }

  const Request* request_;
  Response* response_;
};

// Handler base class for an RPC with a streaming response.
template <typename RequestProto, typename ResponseProto>
class StreamServerResponseHandler
    : public HandlerBase,
      public grpc::ServerWriteReactor<ResponseProto> {
 public:
  using Request = RequestProto;
  using Response = ResponseProto;
  using Reactor = typename grpc::ServerWriteReactor<ResponseProto>;

  StreamServerResponseHandler(::grpc::CallbackServerContext* grpc_context,
                              const Request* request)
      : HandlerBase(grpc_context), request_(request) {}

  using Reactor::Finish;
  void Finish(absl::Status status) {
    Finish(tensorstore::internal::AbslStatusToGrpcStatus(status));
  }

  const Request* request() { return request_; }

 protected:
  void OnDone() final { auto adopted = Adopt(); }

  const Request* request_;
};

// Handler base class for an RPC with a streaming request.
template <typename RequestProto, typename ResponseProto>
class StreamClientRequestHandler
    : public HandlerBase,
      public grpc::ServerReadReactor<RequestProto> {
 public:
  using Request = RequestProto;
  using Response = ResponseProto;
  using Reactor = typename grpc::ServerReadReactor<RequestProto>;

  StreamClientRequestHandler(::grpc::CallbackServerContext* grpc_context,
                             Response* response)
      : HandlerBase(grpc_context), response_(response) {}

  using Reactor::Finish;
  void Finish(absl::Status status) {
    Finish(tensorstore::internal::AbslStatusToGrpcStatus(status));
  }

  Response* response() { return response_; }

 protected:
  void OnDone() final { auto adopted = Adopt(); }

  Response* response_;
};

}  // namespace tensorstore_grpc

#endif  // TENSORSTORE_KVSTORE_TSGRPC_HANDLER_TEMPLATE_H_
