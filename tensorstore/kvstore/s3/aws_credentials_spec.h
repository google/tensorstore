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

#ifndef TENSORSTORE_KVSTORE_S3_AWS_CREDENTIALS_SPEC_H_
#define TENSORSTORE_KVSTORE_S3_AWS_CREDENTIALS_SPEC_H_

#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Spec for AwsCredentialsResource resource.
///
/// Supported spec examples:
///
///   { "type": "default",
///     "profile": "my-profile" }
///
///   { "type": "anonymous" }
///
///   { "type": "environment" }
///
///   { "type": "imds" }
///
///   { "type": "profile",
///     "profile": "my-profile",
///     "config_file": "~/.aws/config",
///     "credentials_file": "~/.aws/credentials" }
///
///   { "type": "ecs",
///     "endpoint": "http://169.254.169.254/latest/meta-data/iam/",
///     "auth_token_file": "/var/run/secrets/aws-ecs/token",
///     "auth_token": "my-auth-token" }
///
class AwsCredentialsSpec final {
 public:
  struct Default final {
    std::string profile;
    friend bool operator==(const Default& a, const Default& b) {
      return a.profile == b.profile;
    }
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.profile);
    };
  };

  struct Anonymous final {
    friend bool operator==(const Anonymous& a, const Anonymous& b) {
      return true;
    }
  };

  struct Environment final {
    friend bool operator==(const Environment& a, const Environment& b) {
      return true;
    }
  };

  struct Imds final {
    friend bool operator==(const Imds& a, const Imds& b) { return true; }
  };

  struct Profile final {
    std::string profile;
    std::string config_file;
    std::string credentials_file;

    friend bool operator==(const Profile& a, const Profile& b) {
      return std::tie(a.profile, a.config_file, a.credentials_file) ==
             std::tie(b.profile, b.config_file, b.credentials_file);
    }
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.profile, x.config_file, x.credentials_file);
    };
  };

  struct EcsRole final {
    std::string endpoint;
    std::string auth_token_file;
    std::string auth_token;
    friend bool operator==(const EcsRole& a, const EcsRole& b) {
      return std::tie(a.endpoint, a.auth_token_file, a.auth_token) ==
             std::tie(b.endpoint, b.auth_token_file, b.auth_token);
    }
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.endpoint, x.auth_token_file, x.auth_token);
    };
  };

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.config);
  };

  friend bool operator==(const AwsCredentialsSpec& a,
                         const AwsCredentialsSpec& b) {
    return a.config == b.config;
  };

  friend bool operator!=(const AwsCredentialsSpec& a,
                         const AwsCredentialsSpec& b) {
    return !(a == b);
  };

  /// Partial binder for ExperimentalGcsGrpcCredentialsSpec.
  class PartialBinder {
   public:
    absl::Status operator()(std::true_type is_loading,
                            const internal_json_binding::NoOptions& options,
                            AwsCredentialsSpec* value,
                            ::nlohmann::json::object_t* j) const;
    absl::Status operator()(std::false_type is_loading,
                            const tensorstore::IncludeDefaults& options,
                            const AwsCredentialsSpec* value,
                            ::nlohmann::json::object_t* j) const;
  };

  using Config =
      std::variant<Default, Anonymous, Environment, Imds, Profile, EcsRole>;
  Config config;
};

Result<internal_aws::AwsCredentialsProvider> MakeAwsCredentialsProvider(
    const AwsCredentialsSpec& spec);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_AWS_CREDENTIALS_SPEC_H_
