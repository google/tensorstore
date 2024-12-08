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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_USE_DIRECTPATH_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_USE_DIRECTPATH_H_

namespace tensorstore {
namespace internal_gcs_grpc {

/// Returns whether to use the directpath GCS endpoint by default.
bool UseDirectPathGcsEndpointByDefault();

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_USE_DIRECTPATH_H_
