// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_GRPC_UTILS_H_
#define TENSORSTORE_INTERNAL_GRPC_UTILS_H_

#include <grpcpp/support/status.h>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal {

/// Converts a grpc::Status to an absl::Status
absl::Status GrpcStatusToAbslStatus(grpc::Status s);

/// Converts an absl::Status to a grpc::Status
grpc::Status AbslStatusToGrpcStatus(const absl::Status& status);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_UTILS_H_
