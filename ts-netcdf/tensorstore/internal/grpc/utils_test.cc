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

#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace {

using ::tensorstore::internal::AbslStatusToGrpcStatus;
using ::tensorstore::internal::GrpcStatusToAbslStatus;

TEST(StatusToGrpcStatus, Basic) {
  EXPECT_EQ(grpc::Status::OK.error_code(),
            AbslStatusToGrpcStatus(absl::OkStatus()).error_code());
}

TEST(GrpcStatusToStatus, Basic) {
  EXPECT_EQ(absl::OkStatus(), GrpcStatusToAbslStatus(grpc::Status::OK));
}

}  // namespace
