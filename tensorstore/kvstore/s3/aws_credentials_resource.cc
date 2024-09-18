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

#include "tensorstore/kvstore/s3/aws_credentials_resource.h"

#include <stddef.h>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/s3/credentials/aws_credentials.h"
#include "tensorstore/kvstore/s3/credentials/default_credential_provider.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace jb = tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_kvstore_s3 {

using Spec = ::tensorstore::internal_kvstore_s3::AwsCredentialsResource::Spec;
using Resource =
    ::tensorstore::internal_kvstore_s3::AwsCredentialsResource::Resource;

const internal::ContextResourceRegistration<AwsCredentialsResource>
    aws_credentials_registration;

Result<Resource> AwsCredentialsResource::Create(
    const Spec& spec, internal::ContextResourceCreationContext context) const {
  if (spec.anonymous) {
    return Resource{spec, nullptr};
  }
  auto result = GetAwsCredentialProvider(
      spec.filename, spec.profile, spec.metadata_endpoint,
      internal_http::GetDefaultHttpTransport());
  if (!result.ok() && absl::IsNotFound(result.status())) {
    return Resource{spec, nullptr};
  }
  TENSORSTORE_RETURN_IF_ERROR(result);
  return Resource{spec, *std::move(result)};
}

Result<std::optional<AwsCredentials>>
AwsCredentialsResource::Resource::GetCredentials() {
  if (!credential_provider_) return std::nullopt;
  auto credential_result_ = credential_provider_->GetCredentials();
  if (!credential_result_.ok() &&
      absl::IsNotFound(credential_result_.status())) {
    return std::nullopt;
  }
  return credential_result_;
}

namespace {

static constexpr auto kAnonymousBinder = jb::Object(jb::Member(
    "anonymous", jb::Projection<&Spec::anonymous>(
                     jb::Validate([](const auto& options, bool* x) {
                       if (*x != true) {
                         return absl::InvalidArgumentError(
                             "\"anonymous\" must be true or not present in "
                             "\"aws_credentials\"");
                       }
                       return absl::OkStatus();
                     }))));

static constexpr auto kParameterBinder = jb::Object(
    jb::OptionalMember("profile", jb::Projection<&Spec::profile>()),
    jb::OptionalMember("filename", jb::Projection<&Spec::filename>()),
    jb::OptionalMember("metadata_endpoint",
                       jb::Projection<&Spec::metadata_endpoint>()));

}  // namespace

/* static */
absl::Status AwsCredentialsResource::FromJsonImpl(
    const JsonSerializationOptions& options, Spec* spec, ::nlohmann::json* j) {
  if (auto* j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
      j_obj && j_obj->find("anonymous") != j_obj->end()) {
    return kAnonymousBinder(std::true_type{}, options, spec, j);
  }
  return kParameterBinder(std::true_type{}, options, spec, j);
}

/* static*/
absl::Status AwsCredentialsResource::ToJsonImpl(
    const JsonSerializationOptions& options, const Spec* spec,
    ::nlohmann::json* j) {
  if (spec->anonymous) {
    return kAnonymousBinder(std::false_type{}, options, spec, j);
  }
  return kParameterBinder(std::false_type{}, options, spec, j);
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
