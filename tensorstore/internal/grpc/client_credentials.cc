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

#include "tensorstore/internal/grpc/client_credentials.h"

#include <memory>
#include <utility>

// Placeholder for internal channel credential  // net
// Placeholder for internal credential options  // net
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace {

ABSL_CONST_INIT static absl::Mutex credentials_mu(absl::kConstInit);

const internal::ContextResourceRegistration<GrpcClientCredentials>
    grpc_client_credentials_registration;

}  // namespace

// TODO: We should extend this class to permit, at least, some selection
// of grpc credentials. See grpcpp/security/credentials.h for options, such as:
//   ::grpc::experimental::LocalCredentials(LOCAL_TCP)
//   ::grpc::GoogleDefaultCredentials();

/*  static*/
bool GrpcClientCredentials::Use(
    tensorstore::Context context,
    std::shared_ptr<::grpc::ChannelCredentials> credentials) {
  auto resource = context.GetResource<GrpcClientCredentials>().value();
  absl::MutexLock l(&credentials_mu);
  bool result = (resource->credentials_ == nullptr);
  resource->credentials_ = std::move(credentials);
  return result;
}

std::shared_ptr<::grpc::ChannelCredentials>
GrpcClientCredentials::Resource::GetCredentials() {
  {
    absl::MutexLock l(&credentials_mu);
    if (credentials_) return credentials_;
  }

  return grpc::InsecureChannelCredentials();
}

}  // namespace tensorstore
