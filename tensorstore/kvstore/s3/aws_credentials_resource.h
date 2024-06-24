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
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Specifies the AWS profile name.
/// TODO: Allow more complex credential specification, which could be any of:
///  {"profile", "filename"} or { "access_key", "secret_key", "session_token" }
///
struct AwsCredentialsResource
    : public internal::ContextResourceTraits<AwsCredentialsResource> {
  static constexpr char id[] = "aws_credentials";

  struct Spec {
    std::string profile;
    std::string filename;
    std::string metadata_endpoint;
    bool anonymous = false;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.profile, x.filename, x.metadata_endpoint, x.anonymous);
    };
  };

  struct Resource {
    Spec spec;
    std::shared_ptr<AwsCredentialProvider> credential_provider_;

    Result<std::optional<AwsCredentials>> GetCredentials();
  };

  static Spec Default() { return Spec{}; }

  static constexpr auto JsonBinder() {
    return [](auto is_loading, const auto& options, auto* obj,
              auto* j) -> absl::Status {
      if constexpr (is_loading) {
        return AwsCredentialsResource::FromJsonImpl(options, obj, j);
      } else {
        return AwsCredentialsResource::ToJsonImpl(options, obj, j);
      }
    };
  }

  Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) const;

  Spec GetSpec(const Resource& resource,
               const internal::ContextSpecBuilder& builder) const {
    return resource.spec;
  }

 private:
  static absl::Status FromJsonImpl(const JsonSerializationOptions& options,
                                   Spec* spec, ::nlohmann::json* j);

  static absl::Status ToJsonImpl(const JsonSerializationOptions& options,
                                 const Spec* spec, ::nlohmann::json* j);
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_CREDENTIALS_AWS_CREDENTIALS_RESOURCE_H_
