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

#include "tensorstore/internal/grpc/utils.h"

#include <grpcpp/support/status.h>

#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "grpcpp/support/status.h"  // third_party
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status.h"

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
namespace {

template <typename T, typename = void>
struct HasAbslStatusOperator : std::false_type {};

template <typename T>
struct HasAbslStatusOperator<
    T, std::void_t<decltype(std::declval<const T&>().operator absl::Status())>>
    : std::true_type {};

// Overload for when operator exists
template <typename T,
          std::enable_if_t<HasAbslStatusOperator<T>::value, int> = 0>
absl::Status ToAbslStatusImpl(const T& s) {
  return s.operator absl::Status();
}

// Overload for when operator does not exist
template <typename T,
          std::enable_if_t<!HasAbslStatusOperator<T>::value, int> = 0>
absl::Status ToAbslStatusImpl(const T& s) {
  auto absl_code = static_cast<absl::StatusCode>(s.error_code());
  absl::Status status(absl_code, s.error_message());
  if (!s.error_details().empty()) {
    // NOTE: Is error_details() a serialized protobuf::Any?
    status.SetPayload("grpc.Status.details", absl::Cord(s.error_details()));
  }
  return status;
}

}  // namespace

absl::Status GrpcStatusToAbslStatus(grpc::Status s, SourceLocation loc) {
  if (s.ok()) return absl::OkStatus();
  absl::Status status = ToAbslStatusImpl(s);
  MaybeAddSourceLocation(status, loc);
  return status;
}

grpc::Status AbslStatusToGrpcStatus(const absl::Status& status) {
  if (status.ok()) return grpc::Status::OK;

  auto grpc_code = static_cast<grpc::StatusCode>(status.code());
  return grpc::Status(grpc_code, std::string(status.message()));
}

}  // namespace internal
}  // namespace tensorstore
