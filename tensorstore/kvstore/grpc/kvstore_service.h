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

#ifndef TENSORSTORE_KVSTORE_GRPC_KVSTORE_SERVICE_H_
#define TENSORSTORE_KVSTORE_GRPC_KVSTORE_SERVICE_H_

#include "grpcpp/server_context.h"  // third_party
#include "grpcpp/support/server_callback.h"  // third_party
#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/grpc/kvstore.pb.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore_grpc {

/// Default forwarding implementation of tensorstore_grpc::KvStoreService.
class KvStoreServiceImpl final
    : public kvstore::grpc_gen::KvStoreService::CallbackService {
 public:
  KvStoreServiceImpl(tensorstore::KvStore kvstore);

  ::grpc::ServerUnaryReactor* Read(
      ::grpc::CallbackServerContext* /*context*/,
      const ::tensorstore_grpc::kvstore::ReadRequest* request,
      ::tensorstore_grpc::kvstore::ReadResponse* response) override;

  ::grpc::ServerUnaryReactor* Write(
      ::grpc::CallbackServerContext* /*context*/,
      const ::tensorstore_grpc::kvstore::WriteRequest* request,
      ::tensorstore_grpc::kvstore::WriteResponse* response) override;

  ::grpc::ServerUnaryReactor* Delete(
      ::grpc::CallbackServerContext* /*context*/,
      const ::tensorstore_grpc::kvstore::DeleteRequest* request,
      ::tensorstore_grpc::kvstore::DeleteResponse* response) override;

  ::grpc::ServerWriteReactor< ::tensorstore_grpc::kvstore::ListResponse>* List(
      ::grpc::CallbackServerContext* /*context*/,
      const ::tensorstore_grpc::kvstore::ListRequest* request) override;

  // Accessor
  tensorstore::KvStore kvstore() { return kvstore_; }

 private:
  tensorstore::KvStore kvstore_;
};

}  // namespace tensorstore_grpc

#endif  // TENSORSTORE_KVSTORE_GRPC_KVSTORE_SERVICE_H_
