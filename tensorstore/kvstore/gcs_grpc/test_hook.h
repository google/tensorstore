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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_DEBUG_HOOK_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_DEBUG_HOOK_H_

#include "absl/functional/any_invocable.h"
#include "grpcpp/client_context.h"  // third_party

// proto
#include "google/storage/v2/storage.pb.h"

namespace tensorstore {
namespace internal_gcs_grpc {

void SetTestHook(
    absl::AnyInvocable<void(grpc::ClientContext*,
                            google::storage::v2::ReadObjectRequest&)>&&
        debug_hook);
void SetTestHook(
    absl::AnyInvocable<void(grpc::ClientContext*,
                            google::storage::v2::WriteObjectRequest&)>&&
        debug_hook);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_DEBUG_HOOK_H_
