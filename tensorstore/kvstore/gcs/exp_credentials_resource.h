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

#ifndef TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_RESOURCE_H_
#define TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_RESOURCE_H_

#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/gcs/exp_credentials_spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// Context resource for gcs_grpc credentials  allows setting non-default
/// credentials for the gcs_grpc kvstore driver, such as external_account
/// credentials.
/// NOTE: This is experimental and may change without notice.
///
/// See exp_credentials_spec.h for supported spec examples.
struct ExperimentalGcsGrpcCredentials final
    : public internal::ContextResourceTraits<ExperimentalGcsGrpcCredentials> {
  static constexpr char id[] = "experimental_gcs_grpc_credentials";
  constexpr static bool config_only = true;

  using Spec = ExperimentalGcsGrpcCredentialsSpec;
  using Resource = Spec;

  static Spec Default() { return Spec{}; }
  static constexpr auto JsonBinder() {
    return internal_json_binding::Object(Spec::PartialBinder{});
  }

  static Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) {
    return spec;
  }

  static Spec GetSpec(const Resource& resource,
                      const internal::ContextSpecBuilder& builder) {
    return resource;
  }
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_RESOURCE_H_
