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

#include "tensorstore/kvstore/gcs/exp_credentials_spec.h"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/grpc/clientauth/authentication_strategy.h"
#include "tensorstore/internal/grpc/clientauth/call_authentication.h"
#include "tensorstore/internal/grpc/clientauth/channel_authentication.h"
#include "tensorstore/internal/grpc/clientauth/impersonate_service_account.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep

namespace jb = ::tensorstore::internal_json_binding;

using ::tensorstore::internal_grpc::CaInfo;
using ::tensorstore::internal_os::ReadAllToString;
using Spec =
    ::tensorstore::internal_storage_gcs::ExperimentalGcsGrpcCredentialsSpec;

namespace tensorstore {
namespace internal_storage_gcs {

struct ExperimentalGcsGrpcCredentialsSpec::Access {
  using Config = Spec::Config;
  Config& operator()(Spec& spec) { return spec.config; }
  const Config& operator()(const Spec& spec) { return spec.config; }
  Config& operator()(Spec* spec) { return spec->config; }
  const Config& operator()(const Spec* spec) { return spec->config; }
};

namespace {

// Returns true if the credential is one of the known credentials.
bool IsKnownCredential(std::string_view credential) {
  for (auto v : {
           "insecure",
           "google_default",
           "access_token",
           "service_account",
           "external_account",
           "impersonate_service_account",
       }) {
    if (v == credential) {
      return true;
    }
  }
  return false;
}

template <typename L, typename T>
using MaybeConst =
    std::conditional_t<std::is_same_v<L, std::true_type>,
                       internal::type_identity_t<T>, std::add_const_t<T>>;

// The PartialBinder for ExperimentalGcsGrpcCredentialsSpec looks at the
// "type" member and then invokes the credential-specific binder.
const auto kPartialBinder = [](auto is_loading, const auto& options, auto* obj,
                               nlohmann::json::object_t* j) {
  // This uses a custom variant binder, so extract the credential type
  // first before using std::visit for the credential-specific binding.
  std::string credentials;
  if constexpr (!is_loading) {
    credentials = obj->GetType();
  }

  TENSORSTORE_RETURN_IF_ERROR(jb::Member(
      "type", jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())(
      is_loading, options, &credentials, j));

  if (!credentials.empty() && !IsKnownCredential(credentials)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid credentials : ", QuoteString(credentials)));
  }

  Spec::Access config;

  if constexpr (is_loading) {
    // On load, install the default value for the loading credential type
    // so that std::visit will invoke the correct binding.
    if (credentials == "access_token") {
      config(obj) = Spec::AccessToken{};
    } else if (credentials == "service_account") {
      config(obj) = Spec::ServiceAccount{};
    } else if (credentials == "external_account") {
      config(obj) = Spec::ExternalAccount{};
    } else if (credentials == "impersonate_service_account") {
      config(obj) = Spec::ImpersonateServiceAccount{};
    } else {
      config(obj) = credentials;
    }
  }

  using L = decltype(is_loading);
  using O = decltype(options);

  // Invoke the binding for the credential type.  Note that MaybeConst<L, T>
  // is used to ensure that the binding is invoked with a const T& when saving
  // to satisfy the constraints of the std::visit call.

  struct BindingVisitor {
    L is_loading;
    O& options;
    nlohmann::json::object_t* j;

    absl::Status operator()(const std::string& v) { return absl::OkStatus(); }
    absl::Status operator()(MaybeConst<L, Spec::AccessToken>& spec) {
      return jb::Member("access_token",
                        jb::Projection<&Spec::AccessToken::access_token>())(
          is_loading, options, &spec, j);
    }
    absl::Status operator()(MaybeConst<L, Spec::ServiceAccount>& spec) {
      bool clear = false;
      if constexpr (std::is_same_v<L, std::true_type>) {
        // Loading.
        if (j->count("path") == 0) {
          spec.json = *j;
          spec.json["type"] = "service_account";
          clear = true;
        }
      } else {
        if (spec.path.empty()) {
          *j = spec.json;
        }
      }
      TENSORSTORE_RETURN_IF_ERROR(jb::OptionalMember(
          "path",
          jb::Projection<&Spec::ServiceAccount::path>(
              jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()))(
          is_loading, options, &spec, j));

      if constexpr (std::is_same_v<L, std::true_type>) {
        if (clear) j->clear();
      }
      return absl::OkStatus();
    }
    absl::Status operator()(MaybeConst<L, Spec::ExternalAccount>& spec) {
      bool clear = false;
      if constexpr (std::is_same_v<L, std::true_type>) {
        // Loading.
        if (j->count("path") == 0) {
          spec.json = *j;
          spec.json.erase("scopes");
          spec.json["type"] = "external_account";
          clear = true;
        }
      } else {
        if (spec.path.empty()) {
          *j = spec.json;
        }
      }

      TENSORSTORE_RETURN_IF_ERROR(jb::Sequence(
          jb::OptionalMember(
              "scopes",
              jb::Projection<&Spec::ExternalAccount::scopes>(
                  jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())),
          jb::OptionalMember(
              "path",
              jb::Projection<&Spec::ExternalAccount::path>(
                  jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())))(
          is_loading, options, &spec, j));

      if constexpr (std::is_same_v<L, std::true_type>) {
        if (clear) j->clear();
      }
      return absl::OkStatus();
    }

    absl::Status operator()(
        MaybeConst<L, Spec::ImpersonateServiceAccount>& spec) {
      return jb::Validate(
          [](const auto&, auto* spec) {
            if (spec->base.empty()) {
              return absl::InvalidArgumentError(
                  "ImpersonateServiceAccount must have a base credential");
            }
            return absl::OkStatus();
          },
          jb::Sequence(
              jb::Member("target_service_account",
                         jb::Projection<&Spec::ImpersonateServiceAccount::
                                            target_service_account>()),
              jb::OptionalMember(
                  "scopes",
                  jb::Projection<&Spec::ImpersonateServiceAccount::scopes>(
                      jb::DefaultInitializedValue<
                          jb::kNeverIncludeDefaults>())),
              jb::OptionalMember(
                  "delegates",
                  jb::Projection<&Spec::ImpersonateServiceAccount::delegates>(
                      jb::DefaultInitializedValue<
                          jb::kNeverIncludeDefaults>())),
              jb::Member("base",
                         jb::Projection<&Spec::ImpersonateServiceAccount::base>(
                             jb::DefaultInitializedValue<
                                 jb::kNeverIncludeDefaults>()))))(
          is_loading, options, &spec, j);
    }
  };

  return std::visit(BindingVisitor{is_loading, options, j}, config(obj));
};

}  // namespace

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    Insecure access_token)
    : config("insecure") {}

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    GoogleDefault access_token)
    : config("google_default") {}

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    AccessToken access_token)
    : config(std::move(access_token)) {}

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    ServiceAccount service_account)
    : config(std::move(service_account)) {}

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    ExternalAccount external_account)
    : config(std::move(external_account)) {}

ExperimentalGcsGrpcCredentialsSpec::ExperimentalGcsGrpcCredentialsSpec(
    ImpersonateServiceAccount impersonate_service_account)
    : config(std::move(impersonate_service_account)) {}

std::string ExperimentalGcsGrpcCredentialsSpec::GetType() const {
  struct TypeVisitor {
    std::string operator()(const std::string& v) { return v; }
    std::string operator()(const AccessToken& spec) { return "access_token"; }
    std::string operator()(const ServiceAccount& spec) {
      return "service_account";
    }
    std::string operator()(const ExternalAccount& spec) {
      return "external_account";
    }
    std::string operator()(const ImpersonateServiceAccount& spec) {
      return "impersonate_service_account";
    }
  };
  return std::visit(TypeVisitor{}, config);
}

absl::Status ExperimentalGcsGrpcCredentialsSpec::PartialBinder::operator()(
    std::true_type is_loading, const internal_json_binding::NoOptions& options,
    Spec* value, ::nlohmann::json::object_t* j) const {
  TENSORSTORE_RETURN_IF_ERROR(kPartialBinder(is_loading, options, value, j));

  if (const auto* impersonate =
          std::get_if<ImpersonateServiceAccount>(&value->config);
      impersonate != nullptr) {
    ::nlohmann::json::object_t j_copy = impersonate->base;
    Spec base;
    return kPartialBinder(is_loading, options, &base, &j_copy);
  }
  return absl::OkStatus();
}
absl::Status ExperimentalGcsGrpcCredentialsSpec::PartialBinder::operator()(
    std::false_type is_loading, const tensorstore::IncludeDefaults& options,
    const Spec* value, ::nlohmann::json::object_t* j) const {
  return kPartialBinder(is_loading, options, value, j);
}

bool Spec::IsDefault() const { return *this == Spec{}; }

Result<std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>>
MakeGrpcAuthenticationStrategy(const Spec& spec, CaInfo ca_info) {
  using R = Result<std::shared_ptr<internal_grpc::GrpcAuthenticationStrategy>>;

  struct Visitor {
    CaInfo ca_info;
    R operator()(const std::string& spec) {
      if (spec.empty()) {
        return nullptr;
      }
      if (spec == "insecure") {
        return internal_grpc::CreateInsecureAuthenticationStrategy();
      }
      if (spec == "google_default") {
        return internal_grpc::CreateGoogleDefaultAuthenticationStrategy();
      }
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Unknown credentials : ", QuoteString(spec)));
    }

    R operator()(const Spec::AccessToken& spec) {
      return internal_grpc::CreateAccessTokenAuthenticationStrategy(
          spec.access_token, ca_info);
    }
    R operator()(const Spec::ServiceAccount& spec) {
      std::string json_string;
      if (!spec.path.empty()) {
        TENSORSTORE_ASSIGN_OR_RETURN(json_string, ReadAllToString(spec.path));
      } else if (!spec.json.empty()) {
        json_string = ::nlohmann::json(spec.json).dump();
      } else {
        return absl::InvalidArgumentError(
            "ServiceAccount must have either a path or a json config");
      }
      return internal_grpc::CreateServiceAccountAuthenticationStrategy(
          json_string, ca_info);
    }
    R operator()(const Spec::ExternalAccount& spec) {
      std::string json_string;
      if (!spec.path.empty()) {
        TENSORSTORE_ASSIGN_OR_RETURN(json_string, ReadAllToString(spec.path));
      } else if (!spec.json.empty()) {
        json_string = ::nlohmann::json(spec.json).dump();
      } else {
        return absl::InvalidArgumentError(
            "ExternalAccount must have either a path or a json config");
      }

      return internal_grpc::CreateExternalAccountAuthenticationStrategy(
          json_string, spec.scopes, ca_info);
    }
    R operator()(const Spec::ImpersonateServiceAccount& spec) {
      if (spec.base.empty()) {
        return absl::InvalidArgumentError(
            "ImpersonateServiceAccount must have a base credential");
      }
      ::nlohmann::json::object_t j_copy = spec.base;
      Spec base;
      TENSORSTORE_RETURN_IF_ERROR(
          kPartialBinder(std::true_type{}, jb::NoOptions{}, &base, &j_copy));
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto base_strategy, MakeGrpcAuthenticationStrategy(base, ca_info));

      internal_grpc::ImpersonateServiceAccountConfig config;
      config.target_service_account = spec.target_service_account;
      config.scopes = spec.scopes;
      config.delegates = spec.delegates;

      return internal_grpc::GrpcImpersonateServiceAccount::Create(
          config, ca_info, std::move(base_strategy));
    }
  };

  Spec::Access config;
  return std::visit(Visitor{ca_info}, config(spec));
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
