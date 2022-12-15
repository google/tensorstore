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

#include <grpcpp/support/status.h>

#include <string>

#include "absl/status/status.h"

// Verify that the grpc and absl status codes are equivalent.

#define TENSORSTORE_STATUS_ASSERT(x, y)                  \
  static_assert(static_cast<int>(grpc::StatusCode::x) == \
                static_cast<int>(absl::StatusCode::y))

TENSORSTORE_STATUS_ASSERT(CANCELLED, kCancelled);
TENSORSTORE_STATUS_ASSERT(UNKNOWN, kUnknown);
TENSORSTORE_STATUS_ASSERT(INVALID_ARGUMENT, kInvalidArgument);
TENSORSTORE_STATUS_ASSERT(DEADLINE_EXCEEDED, kDeadlineExceeded);
TENSORSTORE_STATUS_ASSERT(NOT_FOUND, kNotFound);
TENSORSTORE_STATUS_ASSERT(ALREADY_EXISTS, kAlreadyExists);
TENSORSTORE_STATUS_ASSERT(PERMISSION_DENIED, kPermissionDenied);
TENSORSTORE_STATUS_ASSERT(RESOURCE_EXHAUSTED, kResourceExhausted);
TENSORSTORE_STATUS_ASSERT(FAILED_PRECONDITION, kFailedPrecondition);
TENSORSTORE_STATUS_ASSERT(ABORTED, kAborted);
TENSORSTORE_STATUS_ASSERT(OUT_OF_RANGE, kOutOfRange);
TENSORSTORE_STATUS_ASSERT(UNIMPLEMENTED, kUnimplemented);
TENSORSTORE_STATUS_ASSERT(INTERNAL, kInternal);
TENSORSTORE_STATUS_ASSERT(UNAVAILABLE, kUnavailable);
TENSORSTORE_STATUS_ASSERT(DATA_LOSS, kDataLoss);
TENSORSTORE_STATUS_ASSERT(UNAUTHENTICATED, kUnauthenticated);

#undef TENSORSTORE_STATUS_ASSERT

namespace tensorstore {
namespace internal {

absl::Status GrpcStatusToAbslStatus(grpc::Status s) {
  if (s.ok()) return absl::OkStatus();

  auto absl_code = static_cast<absl::StatusCode>(s.error_code());
  return absl::Status(absl_code, s.error_message());
}

grpc::Status AbslStatusToGrpcStatus(const absl::Status& status) {
  if (status.ok()) return grpc::Status::OK;

  auto grpc_code = static_cast<grpc::StatusCode>(status.code());
  return grpc::Status(grpc_code, std::string(status.message()));
}

}  // namespace internal
}  // namespace tensorstore
