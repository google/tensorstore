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

#include <memory>
#include <string>
#include <type_traits>

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/server_credentials.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/kvstore/grpc/kvstore_server.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"

// $ bazel run :kvstore_service_main
//
using ::tensorstore::grpc_kvstore::KvStoreServer;

KvStoreServer::Spec DefaultSpec() {
  return KvStoreServer::Spec::FromJson(  //
             {
                 {"bind_addresses", {"[::]:0"}},
                 {"base", "memory://"},
             })
      .value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<KvStoreServer::Spec>, spec, DefaultSpec(),
          "KvStoreServer spec.");

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);

  tensorstore::Context ctx = tensorstore::Context::Default();

  // Install LOCAL_TCP credentials for this process:
  tensorstore::GrpcServerCredentials::Use(
      ctx, grpc::experimental::LocalServerCredentials(LOCAL_TCP));
  tensorstore::GrpcClientCredentials::Use(
      ctx, grpc::experimental::LocalCredentials(LOCAL_TCP));

  auto server = KvStoreServer::Start({absl::GetFlag(FLAGS_spec).value}, ctx);
  if (!server.ok()) {
    ABSL_LOG(INFO) << "Failed to start KvStoreServer:" << server.status();
    return 2;
  }

  server->Wait();
  return 0;
}
