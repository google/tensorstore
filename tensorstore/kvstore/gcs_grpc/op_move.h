// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_OP_MOVE_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_OP_MOVE_H_

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_gcs_grpc {

class GcsGrpcKeyValueStore;

/// Custom object metadata key which records the write id of the temporary
/// object written by the atomic zonal write path.  `MoveObject` preserves
/// object metadata, so the id identifies which write produced the destination
/// object.
inline constexpr std::string_view kWriteIdMetadataKey = "tensorstore-write-id";

/// Initiates an atomic MoveObject RPC.
///
/// If a retried `MoveObject` fails with `NOT_FOUND` the outcome of the earlier
/// attempt is ambiguous; the destination object is then queried and the move
/// is reported as successful only if the destination carries
/// `kWriteIdMetadataKey == write_id`.  Otherwise, conditional moves return
/// `StorageGeneration::Unknown()` so that callers re-read, while unconditional
/// moves fail with an error.
///
/// \param driver The GcsGrpcKeyValueStore instance.
/// \param source_object The source object name.
/// \param destination_object The destination object name.
/// \param options Write options (applied as preconditions to destination).
/// \param source_generation Optional generation match for source object.
/// \param write_id Write id stored in the source object's metadata, also used
///     as the idempotency token.
Future<TimestampedStorageGeneration> InitiateMoveObject(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    std::string source_object, std::string destination_object,
    kvstore::WriteOptions options, StorageGeneration source_generation,
    std::string write_id);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_OP_MOVE_H_
