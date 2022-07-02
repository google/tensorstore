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

#ifndef TENSORSTORE_PROTO_ENCODE_TIME_H_
#define TENSORSTORE_PROTO_ENCODE_TIME_H_

#include "google/protobuf/timestamp.pb.h"
#include "absl/time/time.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Encodes an absl::Time into a google.protobuf.Timestamp
void AbslTimeToProto(absl::Time t, google::protobuf::Timestamp* proto);

/// Decodes a google.protobuf.Timestamp-compatible pb into an absl::Time.
Result<absl::Time> ProtoToAbslTime(const google::protobuf::Timestamp& proto);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_ENCODE_TIME_H_
