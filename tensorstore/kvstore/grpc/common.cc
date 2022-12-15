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

#include "tensorstore/kvstore/grpc/common.h"

#include <string>

#include "absl/status/status.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/grpc/common.pb.h"
#include "tensorstore/proto/encode_time.h"
#include "tensorstore/util/result.h"

namespace tensorstore_grpc {

using ::tensorstore::internal::AbslTimeToProto;
using ::tensorstore::internal::ProtoToAbslTime;

absl::Status GetMessageStatus(const StatusMessage& t) {
  return absl::Status(static_cast<absl::StatusCode>(t.code()), t.message());
}

void EncodeGenerationAndTimestamp(
    const tensorstore::TimestampedStorageGeneration& gen,
    GenerationAndTimestamp* generation_and_timestamp) {
  AbslTimeToProto(gen.time, generation_and_timestamp->mutable_timestamp());
  generation_and_timestamp->set_generation(gen.generation.value);
}

tensorstore::Result<tensorstore::TimestampedStorageGeneration>
DecodeGenerationAndTimestamp(const GenerationAndTimestamp& t) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto timestamp, ProtoToAbslTime(t.timestamp()));
  return tensorstore::TimestampedStorageGeneration{
      tensorstore::StorageGeneration{t.generation()}, timestamp};
}

}  // namespace tensorstore_grpc
