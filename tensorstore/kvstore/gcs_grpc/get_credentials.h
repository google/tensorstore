// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_GET_CREDENTIALS_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_GET_CREDENTIALS_H_

#include <functional>
#include <memory>
#include <string_view>

#include "grpcpp/security/credentials.h"  // third_party

namespace tensorstore {
namespace internal_gcs_grpc {

// Returns the grpc credentials for the endpoint.
// May set a call_credentials_fn to return call_credentials.
std::shared_ptr<::grpc::ChannelCredentials> GetCredentialsForEndpoint(
    std::string_view endpoint,
    std::function<std::shared_ptr<grpc::CallCredentials>()>&
        call_credentials_fn);

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_GET_CREDENTIALS_H_
