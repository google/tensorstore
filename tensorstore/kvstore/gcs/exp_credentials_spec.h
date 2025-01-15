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

#ifndef TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_SPEC_H_
#define TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_SPEC_H_

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// Spec for the ExperimentalGcsGrpcCredentials resource.
/// NOTE: This is experimental and may change without notice.
///
/// Supported spec examples:
///
///   { "type": "insecure" }
///   { "type": "google_default" }
///   { "type": "access_token" }
///   { "type": "service_account",
///     "path": "/path/to/service_account.json",
///     ... when path is not present, the service account config is
///     ... loaded from the remaining fields.
///   }
///   { "type": "external_account",
///     "scopes": [
///       "https://www.googleapis.com/auth/cloud-platform"
///     ],
///     "path": "/path/to/external_account.json",
///     ... when path is not present, the external account config is
///     ... loaded from the remaining fields.
///   }
///   { "type": "impersonate_service_account",
///     "target_service_account": "my-service-account@gserviceaccount.com",
///     "scopes": [
///       "https://www.googleapis.com/auth/cloud-platform"
///     ],
///     "delegates": [
///       "my-service-account@gserviceaccount.com"
///     ],
///     "base": { "type": ... }
///   }
///
class ExperimentalGcsGrpcCredentialsSpec final {
 public:
  struct Insecure final {};
  struct GoogleDefault final {};
  struct AccessToken final {
    std::string access_token;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.access_token);
    };
    friend bool operator==(const AccessToken& a, const AccessToken& b) {
      return a.access_token == b.access_token;
    };
  };
  struct ServiceAccount final {
    std::string path;
    ::nlohmann::json::object_t json;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.path, x.json);
    };
    friend bool operator==(const ServiceAccount& a, const ServiceAccount& b) {
      return a.path == b.path && a.json == b.json;
    };
  };
  struct ExternalAccount final {
    std::string path;
    std::vector<std::string> scopes;
    ::nlohmann::json::object_t json;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.path, x.scopes, x.json);
    };
    friend bool operator==(const ExternalAccount& a, const ExternalAccount& b) {
      return a.path == b.path && a.scopes == b.scopes && a.json == b.json;
    };
  };
  struct ImpersonateServiceAccount final {
    std::string target_service_account;
    std::vector<std::string> scopes;
    std::vector<std::string> delegates;
    ::nlohmann::json::object_t base;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.scopes, x.delegates, x.base);
    };
    friend bool operator==(const ImpersonateServiceAccount& a,
                           const ImpersonateServiceAccount& b) {
      return std::tie(a.target_service_account, a.scopes, a.delegates,
                      a.base) ==
             std::tie(b.target_service_account, b.scopes, b.delegates, b.base);
    };
  };

  ExperimentalGcsGrpcCredentialsSpec() = default;
  ~ExperimentalGcsGrpcCredentialsSpec() = default;

  explicit ExperimentalGcsGrpcCredentialsSpec(Insecure access_token);
  explicit ExperimentalGcsGrpcCredentialsSpec(GoogleDefault access_token);
  explicit ExperimentalGcsGrpcCredentialsSpec(AccessToken access_token);
  explicit ExperimentalGcsGrpcCredentialsSpec(ServiceAccount service_account);
  explicit ExperimentalGcsGrpcCredentialsSpec(ExternalAccount external_account);
  explicit ExperimentalGcsGrpcCredentialsSpec(
      ImpersonateServiceAccount impersonate_service_account);

  /// Returns true if the spec is the default.
  bool IsDefault() const;

  /// Returns the type of the credential.
  std::string GetType() const;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.config);
  };

  friend bool operator==(const ExperimentalGcsGrpcCredentialsSpec& a,
                         const ExperimentalGcsGrpcCredentialsSpec& b) {
    return a.config == b.config;
  };

  friend bool operator!=(const ExperimentalGcsGrpcCredentialsSpec& a,
                         const ExperimentalGcsGrpcCredentialsSpec& b) {
    return !(a == b);
  };

  /// Partial binder for ExperimentalGcsGrpcCredentialsSpec.
  class PartialBinder {
   public:
    absl::Status operator()(std::true_type is_loading,
                            const internal_json_binding::NoOptions& options,
                            ExperimentalGcsGrpcCredentialsSpec* value,
                            ::nlohmann::json::object_t* j) const;
    absl::Status operator()(std::false_type is_loading,
                            const tensorstore::IncludeDefaults& options,
                            const ExperimentalGcsGrpcCredentialsSpec* value,
                            ::nlohmann::json::object_t* j) const;
  };

  /// Access is used to allow the variant to be used in std::visit.
  struct Access;

 private:
  friend struct Access;
  // The credential implementation is opaque to allow changes to the
  // implementation without breaking the JSON binding.
  using Config = std::variant<std::string, AccessToken, ServiceAccount,
                              ExternalAccount, ImpersonateServiceAccount>;
  Config config;
};

/// Maps credentials to a GrpcAuthenticationStrategy.
Result<std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>>
MakeGrpcAuthenticationStrategy(const ExperimentalGcsGrpcCredentialsSpec& spec,
                               internal_grpc::CaInfo ca_info);

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_EXP_CREDENTIALS_SPEC_H_
