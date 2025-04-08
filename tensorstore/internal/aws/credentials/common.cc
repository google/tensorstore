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

#include "tensorstore/internal/aws/credentials/common.h"

#include <cassert>
#include <string>
#include <string_view>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include <aws/auth/credentials.h>
#include <aws/common/allocator.h>
#include <aws/common/atomics.h>
#include <aws/common/zero.h>
#include "tensorstore/internal/aws/aws_api.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/aws/string_view.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/uri_utils.h"

namespace tensorstore {
namespace internal_aws {
namespace {

inline AwsCredentialsProvider AsProvider(aws_credentials_provider* p) {
  assert(aws_atomic_load_int(&p->ref_count) == 1);
  return AwsCredentialsProvider(p, internal::adopt_object_ref);
}

}  // namespace

AwsCredentialsProvider MakeCache(AwsCredentialsProvider provider) {
  aws_allocator* allocator = GetAwsAllocator();

  aws_credentials_provider_cached_options options;
  AWS_ZERO_STRUCT(options);
  options.source = provider.get();
  return AsProvider(aws_credentials_provider_new_cached(allocator, &options));
}

AwsCredentialsProvider MakeDefault(std::string_view profile_name_override) {
  aws_allocator* allocator = GetAwsAllocator();

  aws_credentials_provider_chain_default_options options;
  AWS_ZERO_STRUCT(options);
  options.bootstrap = GetAwsClientBootstrap();
  options.tls_ctx = GetAwsTlsCtx();
  options.profile_name_override =
      StringViewToAwsByteCursor(profile_name_override);

  // NOTE: This is not yet supported.
  // options.function_table = GetAwsHttpMockingIfEnabled();
  return AsProvider(
      aws_credentials_provider_new_chain_default(allocator, &options));
}

AwsCredentialsProvider MakeDefaultWithAnonymous(
    std::string_view profile_name_override) {
  aws_allocator* allocator = GetAwsAllocator();

  auto default_provider = MakeDefault(profile_name_override);
  auto anonymous_provider = MakeAnonymous();

  aws_credentials_provider* providers[2];
  AWS_ZERO_ARRAY(providers);
  providers[0] = default_provider.get();
  providers[1] = anonymous_provider.get();

  aws_credentials_provider_chain_options options;
  AWS_ZERO_STRUCT(options);
  options.provider_count = 2;
  options.providers = providers;

  return AsProvider(aws_credentials_provider_new_chain(allocator, &options));
}

AwsCredentialsProvider MakeAnonymous() {
  aws_allocator* allocator = GetAwsAllocator();

  aws_credentials_provider_shutdown_options options;
  AWS_ZERO_STRUCT(options);
  return AsProvider(
      aws_credentials_provider_new_anonymous(allocator, &options));
}

AwsCredentialsProvider MakeEnvironment() {
  aws_allocator* allocator = GetAwsAllocator();
  aws_credentials_provider_environment_options options;
  AWS_ZERO_STRUCT(options);
  return AsProvider(
      aws_credentials_provider_new_environment(allocator, &options));
}

AwsCredentialsProvider MakeProfile(std::string_view profile_name_override,
                                   std::string_view credentials_file_override,
                                   std::string_view config_file_override) {
  aws_allocator* allocator = GetAwsAllocator();
  aws_credentials_provider_profile_options options;
  AWS_ZERO_STRUCT(options);
  options.profile_name_override =
      StringViewToAwsByteCursor(profile_name_override);
  options.config_file_name_override =
      StringViewToAwsByteCursor(config_file_override);
  options.credentials_file_name_override =
      StringViewToAwsByteCursor(credentials_file_override);

  options.bootstrap = GetAwsClientBootstrap();
  options.tls_ctx = GetAwsTlsCtx();
  options.function_table = GetAwsHttpMockingIfEnabled();

  return AsProvider(aws_credentials_provider_new_profile(allocator, &options));
}

AwsCredentialsProvider MakeImds() {
  aws_allocator* allocator = GetAwsAllocator();
  aws_credentials_provider_imds_options options;
  AWS_ZERO_STRUCT(options);

  options.bootstrap = GetAwsClientBootstrap();
  options.function_table = GetAwsHttpMockingIfEnabled();

  return AsProvider(aws_credentials_provider_new_imds(allocator, &options));
}

AwsCredentialsProvider MakeEcsRole(std::string_view endpoint,
                                   std::string_view auth_token_file_path,
                                   std::string_view auth_token) {
  aws_allocator* allocator = GetAwsAllocator();
  if (endpoint.empty()) {
    assert(auth_token_file_path.empty());
    assert(auth_token.empty());

    aws_credentials_provider_ecs_environment_options options;
    AWS_ZERO_STRUCT(options);
    options.bootstrap = GetAwsClientBootstrap();
    options.tls_ctx = GetAwsTlsCtx();
    options.function_table = GetAwsHttpMockingIfEnabled();
    return AsProvider(
        aws_credentials_provider_new_ecs_from_environment(allocator, &options));
  }

  auto parsed = internal::ParseGenericUri(endpoint);
  auto host_port = internal::SplitHostPort(parsed.authority);
  if (!host_port) {
    return nullptr;
  }
  if (!parsed.authority.empty()) {
    if (!internal::SplitHostPort(parsed.authority)) {
      return nullptr;
    }
  }

  aws_credentials_provider_ecs_options options;
  AWS_ZERO_STRUCT(options);
  options.host = StringViewToAwsByteCursor(host_port->host);
  if (!host_port->port.empty()) {
    if (!absl::SimpleAtoi(host_port->port, &options.port)) {
      options.port = 0;
    }
  }

  std::string path_and_query = absl::StrCat(parsed.path, parsed.query);
  options.path_and_query = StringViewToAwsByteCursor(path_and_query);

  options.auth_token = StringViewToAwsByteCursor(auth_token);
  options.auth_token_file_path =
      StringViewToAwsByteCursor(auth_token_file_path);

  options.bootstrap = GetAwsClientBootstrap();
  if (parsed.scheme == "https") {
    options.tls_ctx = GetAwsTlsCtx();
  }
  options.function_table = GetAwsHttpMockingIfEnabled();

  return AsProvider(aws_credentials_provider_new_ecs(allocator, &options));
}

}  // namespace internal_aws
}  // namespace tensorstore
