// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_OP_SUPPORT_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_OP_SUPPORT_H_

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal_gcs_grpc {

/// Abseil has a convenient crc32_t type, but it doesn't handle absl::Cord.
inline absl::crc32c_t ComputeCrc32c(const absl::Cord& cord) {
  absl::crc32c_t crc{0};
  for (auto chunk : cord.Chunks()) {
    crc = absl::ExtendCrc32c(crc, chunk);
  }
  return crc;
}

/// Returns whether the absl::Status is a retriable request.
/// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/retry_policy.h
inline bool IsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kResourceExhausted ||
          status.code() == absl::StatusCode::kUnavailable ||
          status.code() == absl::StatusCode::kInternal);
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_OP_SUPPORT_H_
