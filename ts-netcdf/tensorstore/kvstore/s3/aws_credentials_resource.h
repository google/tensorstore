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

#ifndef TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_RESOURCE_H_
#define TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_RESOURCE_H_

#include <stddef.h>

#include <cassert>

#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/s3/aws_credentials_spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Specifies credentials for AWS.
struct AwsCredentialsResource
    : public internal::ContextResourceTraits<AwsCredentialsResource> {
  static constexpr char id[] = "aws_credentials";
  constexpr static bool config_only = true;

  using Spec = AwsCredentialsSpec;
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

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_RESOURCE_H_
