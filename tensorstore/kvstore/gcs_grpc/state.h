// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_STATE_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_STATE_H_

#include <stddef.h>

#include <optional>
#include <utility>
#include <vector>

#include "absl/crc/crc32c.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/result.h"

// proto
#include "google/storage/v2/storage.pb.h"

namespace tensorstore {
namespace internal_gcs_grpc {

/// Implementation details for reading from GCS.
struct ReadState {
 public:
  using Request = google::storage::v2::ReadObjectRequest;
  using Response = google::storage::v2::ReadObjectResponse;

  ReadState(kvstore::ReadOptions options) : options_(std::move(options)) {
    ResetWorkingState();
  }

  void ResetWorkingState() {
    storage_generation_ =
        TimestampedStorageGeneration{StorageGeneration::Unknown(), absl::Now()};
    crc32c_ = std::nullopt;
    chunks_.clear();
  }

  absl::Duration GetLatency() const {
    return absl::Now() - storage_generation_.time;
  }

  void SetupRequest(Request& request);

  Result<kvstore::ReadResult> HandleFinalStatus(absl::Status status);

  absl::Status HandleResponse(Response& response);

 private:
  // Initial state.
  kvstore::ReadOptions options_;

  /// Working state.
  TimestampedStorageGeneration storage_generation_;
  std::optional<absl::crc32c_t> crc32c_;
  std::vector<std::pair<absl::Cord, absl::crc32c_t>> chunks_;
};

/// Implementation details for writing to GCS.
struct WriteState {
 public:
  using Request = google::storage::v2::WriteObjectRequest;
  using Response = google::storage::v2::WriteObjectResponse;

  WriteState(kvstore::WriteOptions options, absl::Cord cord)
      : options_(std::move(options)), value_(std::move(cord)) {
    ResetWorkingState();
  }

  void ResetWorkingState() {
    value_offset_ = 0;
    crc32c_ = absl::crc32c_t{0};
    start_time_ = absl::Now();
  }

  absl::Duration GetLatency() const { return absl::Now() - start_time_; }

  Result<TimestampedStorageGeneration> HandleFinalStatus(absl::Status status,
                                                         Response& response);

  void UpdateRequestForNextWrite(Request& request);

  kvstore::WriteOptions options_;
  absl::Cord value_;
  size_t value_offset_;
  absl::Time start_time_;
  absl::crc32c_t crc32c_;
};

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_STATE_H_
