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

#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/security/server_credentials.h"  // third_party
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/grpc/client_credentials.h"
#include "tensorstore/internal/grpc/server_credentials.h"
#include "tensorstore/kvstore/tsgrpc/kvstore_server.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"

/// Starts a grpc kvstore server

/* Example

bazel run //tensorstore/kvstore/grpc:kvstore_server_main &
bazel run //tensorstore/kvstore:live_kvstore_test \
 -- --kvstore_spec='{ "driver": "tsgrpc_kvstore", "address": "localhost:9833" }'

*/
using ::tensorstore::grpc_kvstore::KvStoreServer;

KvStoreServer::Spec DefaultSpec() {
  return KvStoreServer::Spec::FromJson(  //
             {
                 {"bind_addresses", {"localhost:9833"}},
                 {"base", "memory://"},
             })
      .value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<KvStoreServer::Spec>, spec, DefaultSpec(),
          "KvStoreServer spec for reading data.  See example.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore

  tensorstore::Context context(absl::GetFlag(FLAGS_context_spec).value);

  // Install LOCAL_TCP credentials for this process:
  tensorstore::GrpcServerCredentials::Use(
      context, grpc::experimental::LocalServerCredentials(LOCAL_TCP));
  tensorstore::GrpcClientCredentials::Use(
      context, grpc::experimental::LocalCredentials(LOCAL_TCP));

  auto server =
      KvStoreServer::Start({absl::GetFlag(FLAGS_spec).value}, context);
  if (!server.ok()) {
    ABSL_LOG(INFO) << "Failed to start KvStoreServer:" << server.status();
    return 2;
  }

  server->Wait();
  return 0;
}
