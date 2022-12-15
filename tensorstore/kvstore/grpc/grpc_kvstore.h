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

#ifndef TENSORSTORE_KVSTORE_GRPC_GRPC_KVSTORE_H_
#define TENSORSTORE_KVSTORE_GRPC_GRPC_KVSTORE_H_

#include <memory>
#include <string>

#include "absl/time/time.h"
#include "tensorstore/kvstore/grpc/kvstore.grpc.pb.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Creates a new "grpc_kvstore".
///
/// \param grpc_concurrency Concurrency resource used for grpc operations.
/// \param address Address of service.
/// \param stub A pointer to the stub. Optional.
/// \remarks Setting `stub` is an advanced configuration option that allows
/// mocking the internal service call.
Result<kvstore::DriverPtr> CreateGrpcKvStore(
    std::string address, absl::Duration timeout = absl::InfiniteDuration(),
    std::shared_ptr<
        tensorstore_grpc::kvstore::grpc_gen::KvStoreService::StubInterface>
        stub = nullptr);

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GRPC_GRPC_KVSTORE_H_
