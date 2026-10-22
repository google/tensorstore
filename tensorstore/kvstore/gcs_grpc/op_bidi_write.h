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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_OP_BIDI_WRITE_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_OP_BIDI_WRITE_H_

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_gcs_grpc {

class GcsGrpcKeyValueStore;

// Returns whether `status` indicates that the bucket only accepts appendable
// (zonal) objects, i.e. that it is a ``RAPID`` storage class bucket.
//
// GCS does not currently expose a structured reason for this, so the message
// text is matched against the variants the service has been observed to
// return.  This is the sole signal used to auto-detect zonal buckets; if the
// service rewords these messages, detection silently stops working and writes
// to zonal buckets fail with the underlying error.
bool IsZonalObjectRequiredError(const absl::Status& status);

// Returns whether `status` indicates that the bucket does not support
// appendable objects at all.  Same caveat as `IsZonalObjectRequiredError`.
bool IsAppendableNotSupportedError(const absl::Status& status);

/// Initiates a write using BidiWriteObject.
///
/// Queries `driver->zonal_state()` to determine write strategy:
/// - If `is_zonal() == false`: performs a direct non-appendable write.
/// - If `is_non_atomic()`: performs a direct appendable write to `key`
///   without a temporary object or `MoveObject`.
/// - If `is_zonal() == true`: performs an atomic appendable write by writing to
///   a temporary object and moving it to `key` via `MoveObject`.
/// - If `is_zonal() == std::nullopt`: fallback mode; serializes concurrent
///   writes behind a single non-appendable probe write, updates the driver once
///   the bucket type is determined, and falls back to atomic appendable write
///   mode if zonal objects are required.
///
/// \param driver The GcsGrpcKeyValueStore instance.
/// \param common_metrics Common metrics tracking writes.
/// \param key Object key.
/// \param data Object payload.
/// \param options Write options.
Future<TimestampedStorageGeneration> InitiateBidiWrite(
    internal::IntrusivePtr<GcsGrpcKeyValueStore> driver,
    internal_kvstore::CommonMetrics& common_metrics, kvstore::Key key,
    kvstore::Value data, kvstore::WriteOptions options);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_OP_BIDI_WRITE_H_
