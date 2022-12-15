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

#ifndef TENSORSTORE_INTERNAL_GRPC_SERVER_CREDENTIALS_H_
#define TENSORSTORE_INTERNAL_GRPC_SERVER_CREDENTIALS_H_

#include <memory>

#include "grpcpp/security/server_credentials.h"  // third_party
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Context resource for a `grpc::ServerCredentials`.
///
/// This allows setting non-default credentials for use by grpc servers, such
/// as the grpc_kvstore driver, however since credentials may contain sensitive
/// information such as keys, this must be done outside of the context.
///
/// For example:
///
///     auto context = Context::Default();
///
///     auto creds = grpc::experimental::LocalServerCredentials(LOCAL_TCP);
///     tensorstore::GrpcServerCredentials::Use(context, creds);
///
struct GrpcServerCredentials
    : public internal::ContextResourceTraits<GrpcServerCredentials> {
  static constexpr char id[] = "grpc_server_credentials";

  struct Spec {};

  struct Resource {
    // Returns either the owned credentials or a new default credential.
    std::shared_ptr<::grpc::ServerCredentials> GetCredentials();

   private:
    friend struct GrpcServerCredentials;
    std::shared_ptr<::grpc::ServerCredentials> credentials_;
  };

  static constexpr Spec Default() { return {}; }
  static constexpr auto JsonBinder() { return internal_json_binding::Object(); }

  static Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) {
    return Resource{};
  }

  static Spec GetSpec(const Resource& resource,
                      const internal::ContextSpecBuilder& builder) {
    return Spec{};
  }

  /// Installs the `credentials` into the context.
  /// Returns true when prior credentials were nullptr.
  static bool Use(tensorstore::Context context,
                  std::shared_ptr<::grpc::ServerCredentials> credentials);
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_SERVER_CREDENTIALS_H_
