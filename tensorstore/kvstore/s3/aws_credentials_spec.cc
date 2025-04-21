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

#include "tensorstore/kvstore/s3/aws_credentials_spec.h"

#include <stddef.h>

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <aws/common/error.h>
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/aws/aws_api.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_variant.h"  // IWYU pragma: keep

using Spec = ::tensorstore::internal_kvstore_s3::AwsCredentialsSpec;

using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::MakeAnonymous;
using ::tensorstore::internal_aws::MakeCache;
using ::tensorstore::internal_aws::MakeDefaultWithAnonymous;
using ::tensorstore::internal_aws::MakeEcsRole;
using ::tensorstore::internal_aws::MakeEnvironment;
using ::tensorstore::internal_aws::MakeImds;
using ::tensorstore::internal_aws::MakeProfile;

namespace jb = ::tensorstore::internal_json_binding;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

template <typename L, typename T>
using MaybeConst =
    std::conditional_t<std::is_same_v<L, std::true_type>,
                       internal::type_identity_t<T>, std::add_const_t<T>>;

constexpr auto kDefaultBinder =  //
    jb::Sequence(jb::Member(
        "profile",
        jb::Projection<&Spec::Default::profile>(
            jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())));

constexpr auto kProfileBinder =  //
    jb::Sequence(
        jb::Member(
            "profile",
            jb::Projection<&Spec::Profile::profile>(
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())),
        jb::Member(
            "credentials_file",
            jb::Projection<&Spec::Profile::credentials_file>(
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())),
        jb::Member(
            "config_file",
            jb::Projection<&Spec::Profile::config_file>(
                jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()))
        /**/
    );

constexpr auto kEcsRoleBinder =  //
    jb::Validate(
        [](const auto&, auto* spec) -> absl::Status {
          if (spec->endpoint.empty() && !spec->auth_token_file.empty()) {
            return absl::InvalidArgumentError(
                "EcsRole must specify an endpoint when auth_token_file is "
                "specified.");
          }
          if (!spec->endpoint.empty()) {
            auto parsed = internal::ParseGenericUri(spec->endpoint);
            if (!parsed.authority.empty()) {
              if (!internal::SplitHostPort(parsed.authority)) {
                return absl::InvalidArgumentError(tensorstore::StrCat(
                    "Invalid endpoint: ", QuoteString(spec->endpoint)));
              }
            }
          }
          return absl::OkStatus();
        },
        jb::Sequence(
            jb::Member(
                "endpoint",
                jb::Projection<&Spec::EcsRole::endpoint>(
                    jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>())),
            jb::Member(
                "auth_token_file",
                jb::Projection<&Spec::EcsRole::auth_token_file>(
                    jb::DefaultInitializedValue<jb::kNeverIncludeDefaults>()))
            // NOTE: Disallow setting "auth_token" for now.
            /**/
            ));

// The PartialBinder for AwsCredentialsSpec looks at the
// "type" member and then invokes the credential-specific binder.
const auto kPartialBinder = jb::TaggedVariantBinder<std::string, 6>(
    /*tag_binder=*/jb::Member("type"),
    /*tags=*/
    {"default", "anonymous", "environment", "imds", "profile", "ecs"},
    /*value_binders...=*/
    kDefaultBinder, jb::EmptyBinder, jb::EmptyBinder, jb::EmptyBinder,
    kProfileBinder, kEcsRoleBinder);

}  // namespace

absl::Status AwsCredentialsSpec::PartialBinder::operator()(
    std::true_type is_loading, const internal_json_binding::NoOptions& options,
    Spec* value, ::nlohmann::json::object_t* j) const {
  auto status = kPartialBinder(is_loading, options, &value->config, j);
  return MaybeAnnotateStatus(status, "Failed to parse AWS credentials spec");
}

absl::Status AwsCredentialsSpec::PartialBinder::operator()(
    std::false_type is_loading, const tensorstore::IncludeDefaults& options,
    const Spec* value, ::nlohmann::json::object_t* j) const {
  return kPartialBinder(is_loading, options, &value->config, j);
}

/// Maps the spec to an AWS credentials provider.
Result<AwsCredentialsProvider> MakeAwsCredentialsProvider(const Spec& spec) {
  // Ensure that the AWS API is initialized.
  struct MakeCredentialsVisitor {
    using R = AwsCredentialsProvider;

    R operator()(const Spec::Default& v) {
      return MakeDefaultWithAnonymous(v.profile);
    }
    R operator()(const Spec::Anonymous&) { return MakeAnonymous(); }
    R operator()(const Spec::Environment&) { return MakeEnvironment(); }
    R operator()(const Spec::Imds&) { return MakeImds(); }
    R operator()(const Spec::Profile& v) {
      return MakeProfile(v.profile, v.credentials_file, v.config_file);
    }
    R operator()(const Spec::EcsRole& v) {
      return MakeEcsRole(v.endpoint, v.auth_token_file, v.auth_token);
    }
  };

  (void)internal_aws::GetAwsAllocator();  // Force AWS library initialization.
  AwsCredentialsProvider credentials_provider =
      std::visit(MakeCredentialsVisitor{}, spec.config);

  if (!credentials_provider) {
    int err = aws_last_error();
    return (err == 0)
               ? absl::InternalError("Failed to create credentials provider")
               : absl::InternalError(
                     absl::StrCat("Failed to create credentials provider",
                                  aws_error_debug_str(err)));
  }

  return MakeCache(std::move(credentials_provider));
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
