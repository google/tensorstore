// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs/gcs_testbench.h"

#include <optional>
#include <string>


#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"  // third_party
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/subprocess.h"
#include "tensorstore/proto/parse_text_proto_or_die.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// protos
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"

ABSL_FLAG(std::string, testbench_binary, "",
          "Path to the gcs storage-testbench rest_server");

namespace gcs_testbench {

using ::tensorstore::internal::GrpcStatusToAbslStatus;
using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::Subprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_http::GetDefaultHttpTransport;
using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::transport_test_utils::TryPickUnusedPort;
using ::google::storage::v2::Storage;

StorageTestbench::StorageTestbench()
    : http_port(TryPickUnusedPort().value_or(0)),
      grpc_port(TryPickUnusedPort().value_or(0)) {
  ABSL_CHECK(http_port > 0);
  ABSL_CHECK(grpc_port > 0);
}

std::string StorageTestbench::http_address() {
  return absl::StrFormat("localhost:%d", http_port);
}

std::string StorageTestbench::grpc_address() {
  return absl::StrFormat("localhost:%d", grpc_port);
}

void StorageTestbench::SpawnProcess() {
  if (running) return;
  ABSL_LOG(INFO) << "Spawning testbench: http://" << http_address();

  {
    SubprocessOptions options{absl::GetFlag(FLAGS_testbench_binary),
                              {absl::StrFormat("--port=%d", http_port)}};

    /// TODO: getcwd() so that it can be run from anywhere.
    TENSORSTORE_CHECK_OK_AND_ASSIGN(child, SpawnSubprocess(options));
  }

  /// Wait for the process to fully start.
  for (auto deadline = absl::Now() + absl::Seconds(10);;) {
    // Once the process is running, start a gRPC server on the provided port.
    absl::SleepFor(absl::Milliseconds(200));
    auto start_grpc_future = GetDefaultHttpTransport()->IssueRequest(
        HttpRequestBuilder(
            "GET", absl::StrFormat("http://localhost:%d/start_grpc", http_port))
            .AddQueryParameter("port", absl::StrCat(grpc_port))
            .BuildRequest(),
        absl::Cord(), absl::Seconds(15), absl::Seconds(15));
    if (start_grpc_future.status().ok()) break;
    if (absl::Now() < deadline &&
        absl::IsUnavailable(start_grpc_future.status())) {
      continue;
    }
    // Deadline has expired & there's nothing to show for it.
    TENSORSTORE_CHECK_OK(start_grpc_future.status());
  }

  running = true;
}

void StorageTestbench::CreateBucket(std::string bucket) {
  ABSL_CHECK(running);

  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      grpc_address(), grpc::InsecureChannelCredentials());  // NOLINT

  auto stub = Storage::NewStub(channel);

  grpc::ClientContext client_context;
  google::storage::v2::CreateBucketRequest bucket_request =
      tensorstore::ParseTextProtoOrDie(R"pb(
        parent: 'projects/12345'
        bucket: { location: 'US' storage_class: 'STANDARD' }
        bucket_id: 'bucket'
        predefined_acl: 'publicReadWrite'
        predefined_default_object_acl: 'publicReadWrite'
      )pb");
  bucket_request.set_bucket_id(bucket);

  google::storage::v2::Bucket bucket_response;
  grpc::Status status =
      stub->CreateBucket(&client_context, bucket_request, &bucket_response);
  ABSL_LOG(INFO) << GrpcStatusToAbslStatus(status);
}

}  // namespace gcs_testbench
