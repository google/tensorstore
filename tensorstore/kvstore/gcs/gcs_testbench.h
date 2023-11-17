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

#ifndef TENSORSTORE_KVSTORE_GCS_GCS_TESTBENCH_H_
#define TENSORSTORE_KVSTORE_GCS_GCS_TESTBENCH_H_

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/internal/os/subprocess.h"

namespace gcs_testbench {

/// StorageTestbench runs the storage_testbench binary and returns a port.
class StorageTestbench {
 public:
  StorageTestbench();
  ~StorageTestbench();

  // Spawns the subprocess and returns the grpc address.
  void SpawnProcess();

  // Issues a gRPC CreateBucket request against the testbench.
  static absl::Status CreateBucket(std::string grpc_endpoint,
                                   std::string bucket);

  std::string http_address();
  std::string grpc_address();

  int http_port;
  int grpc_port;
  bool running = false;

  std::optional<tensorstore::internal::Subprocess> child;
};

}  // namespace gcs_testbench

#endif  // TENSORSTORE_KVSTORE_GCS_GCS_TESTBENCH_H_
