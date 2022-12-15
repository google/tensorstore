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

#ifndef TENSORSTORE_KVSTORE_GRPC_COMMON_H_
#define TENSORSTORE_KVSTORE_GRPC_COMMON_H_

#include "google/protobuf/timestamp.pb.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/grpc/common.pb.h"
#include "tensorstore/util/result.h"

namespace tensorstore_grpc {

/// Converts between an absl::Time and a google.protobuf.Timestamp
void EncodeTimestamp(absl::Time t, google::protobuf::Timestamp* proto);
tensorstore::Result<absl::Time> DecodeTimestamp(
    const google::protobuf::Timestamp& proto);

/// Encodes a GenerationAndTimestamp protocol buffer
void EncodeGenerationAndTimestamp(
    const tensorstore::TimestampedStorageGeneration& gen,
    GenerationAndTimestamp* generation_and_timestamp);

template <typename T>
void EncodeGenerationAndTimestamp(
    const tensorstore::TimestampedStorageGeneration& gen, T* proto) {
  EncodeGenerationAndTimestamp(gen, proto->mutable_generation_and_timestamp());
}

/// Encodes a GenerationAndTimestamp protocol buffer
tensorstore::Result<tensorstore::TimestampedStorageGeneration>
DecodeGenerationAndTimestamp(const GenerationAndTimestamp& t);

template <typename T>
tensorstore::Result<tensorstore::TimestampedStorageGeneration>
DecodeGenerationAndTimestamp(const T& t) {
  return DecodeGenerationAndTimestamp(t.generation_and_timestamp());
}

/// Returns an absl::Status when given a tensorstore_gpc::StatuMessage
absl::Status GetMessageStatus(const StatusMessage& t);
template <typename T>

absl::Status GetMessageStatus(const T& t) {
  if (!t.has_status()) return absl::OkStatus();
  return GetMessageStatus(t.status());
}

}  // namespace tensorstore_grpc

#endif  // TENSORSTORE_KVSTORE_GRPC_COMMON_H_
