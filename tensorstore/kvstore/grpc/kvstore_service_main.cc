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
#include "grpcpp/server.h"  // third_party
#include "grpcpp/server_builder.h"  // third_party
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/kvstore/grpc/common.h"
#include "tensorstore/kvstore/grpc/kvstore_service.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"

// $ bazel run :kvstore_service_main
//
using ::tensorstore_grpc::KvStoreServiceImpl;

ABSL_FLAG(std::string, bind_address, "[::1]:43231", "Bind address");

tensorstore::kvstore::Spec DefaultKvStoreSpec() {
  return tensorstore::kvstore::Spec::FromJson(  //
             {
                 {"driver", "memory"},
             })
      .value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          DefaultKvStoreSpec(), "kvstore spec for reading/writing data.");

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);

  auto kv = tensorstore::kvstore::Open(absl::GetFlag(FLAGS_kvstore_spec).value)
                .result();
  if (!kv.ok()) {
    ABSL_LOG(INFO) << "Failed to open kvstore:" << kv.status();
    return 2;
  }
  auto service_impl = std::make_shared<KvStoreServiceImpl>(*kv);

  ABSL_LOG(INFO) << "Starting Service";

  std::string bind_address = absl::GetFlag(FLAGS_bind_address);
  if (bind_address.empty()) {
    bind_address = "[::]:0";
  }

  int listening_port = 0;
  std::unique_ptr<grpc::Server> server;

  {
    auto creds = grpc::InsecureServerCredentials();

    grpc::ServerBuilder builder;
    builder.RegisterService(service_impl.get());

    builder.AddListeningPort(bind_address, creds, &listening_port);

    ABSL_LOG(INFO) << "Listening on " << bind_address << " with port "
                   << listening_port;
    server = builder.BuildAndStart();
  }

  server->Wait();
  return 0;
}
