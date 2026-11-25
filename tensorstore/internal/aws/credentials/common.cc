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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include <aws/auth/auth.h>
#include <aws/auth/credentials.h>
#include <aws/common/allocator.h>
#include <aws/common/atomics.h>
#include <aws/common/error.h>
#include <aws/common/zero.h>
#include "tensorstore/internal/aws/aws_api.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/aws/string_view.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/uri/parse.h"

namespace tensorstore {
namespace internal_aws {
namespace {

inline AwsCredentialsProvider AsProvider(aws_credentials_provider* p) {
  assert(!p || aws_atomic_load_int(&p->ref_count) == 1);
  return AwsCredentialsProvider(p, internal::adopt_object_ref);
}

int SyncFailureErrorCode() {
  int err = aws_last_error();
  return err != AWS_ERROR_SUCCESS
             ? err
             : AWS_AUTH_CREDENTIALS_PROVIDER_DELEGATE_FAILURE;
}

class FirstResolvedState
    : public internal::AtomicReferenceCount<FirstResolvedState> {
 public:
  static AwsCredentialsProvider Make(
      absl::Span<const AwsCredentialsProvider> providers) {
    std::vector<AwsCredentialsProvider> valid;
    valid.reserve(providers.size());
    for (const auto& p : providers) {
      if (p) valid.push_back(p);
    }
    if (valid.empty()) return nullptr;
    if (valid.size() == 1) return std::move(valid[0]);

    internal::IntrusivePtr<FirstResolvedState> state(
        new FirstResolvedState(std::move(valid)));

    aws_credentials_provider_delegate_options options;
    AWS_ZERO_STRUCT(options);
    options.get_credentials = &GetCredentials;
    options.delegate_user_data = state.get();
    options.shutdown_options.shutdown_callback = &OnShutdown;
    options.shutdown_options.shutdown_user_data = state.get();

    aws_credentials_provider* provider =
        aws_credentials_provider_new_delegate(GetAwsAllocator(), &options);
    if (!provider) return nullptr;
    state.release();  // Owned by `OnShutdown`.
    return AsProvider(provider);
  }

 private:
  enum class Mode { kUnresolved, kResolving, kLatched };

  struct PendingQuery {
    aws_on_get_credentials_callback_fn* callback;
    void* user_data;
  };

  explicit FirstResolvedState(std::vector<AwsCredentialsProvider> providers)
      : providers_(std::move(providers)) {}

  // Note: `aws_credentials_provider_new_delegate` invokes `OnShutdown` as soon
  // as the outer provider's refcount hits zero. Any in-flight candidate query
  // holds its own `IntrusivePtr<FirstResolvedState>` so `this` remains valid
  // until the in-flight query finishes.
  static void OnShutdown(void* user_data) {
    internal::IntrusivePtr<FirstResolvedState>(
        static_cast<FirstResolvedState*>(user_data),
        internal::adopt_object_ref);
  }

  static void QueryCurrentProvider(
      internal::IntrusivePtr<FirstResolvedState> self,
      AwsCredentialsProvider provider) {
    FirstResolvedState* raw_self = self.release();
    if (aws_credentials_provider_get_credentials(provider.get(),
                                                 &OnCandidateCredentials,
                                                 raw_self) == AWS_OP_SUCCESS) {
      return;  // `raw_self` adopted by `OnCandidateCredentials`.
    }
    OnCandidateCredentials(nullptr, SyncFailureErrorCode(), raw_self);
  }

  static void OnCandidateCredentials(aws_credentials* credentials,
                                     int error_code, void* user_data) {
    internal::IntrusivePtr<FirstResolvedState> self(
        static_cast<FirstResolvedState*>(user_data),
        internal::adopt_object_ref);
    AwsCredentials creds_ref(credentials);

    std::vector<AwsCredentialsProvider> unused_providers;
    AwsCredentialsProvider next_to_try;
    std::vector<PendingQuery> waiters;
    bool succeeded = false;
    int final_error = error_code;
    {
      absl::MutexLock lock(self->mutex_);
      if (error_code == AWS_ERROR_SUCCESS && credentials != nullptr) {
        succeeded = true;
        self->mode_ = Mode::kLatched;
        self->latched_ = std::move(self->providers_[self->current_index_]);
        unused_providers.swap(self->providers_);
        waiters.swap(self->pending_);
      } else if (self->current_index_ + 1 < self->providers_.size()) {
        ++self->current_index_;
        next_to_try = self->providers_[self->current_index_];
      } else {
        self->mode_ = Mode::kUnresolved;
        self->current_index_ = 0;
        waiters.swap(self->pending_);
        if (final_error == AWS_ERROR_SUCCESS) {
          final_error = AWS_AUTH_CREDENTIALS_PROVIDER_DELEGATE_FAILURE;
        }
      }
    }
    // Release unused providers and invoke callbacks outside `mutex_` so
    // synchronous callbacks or shutdown hooks cannot deadlock.
    unused_providers.clear();

    if (next_to_try) {
      QueryCurrentProvider(std::move(self), std::move(next_to_try));
      return;
    }

    if (succeeded) {
      for (const auto& w : waiters) {
        w.callback(creds_ref.get(), AWS_ERROR_SUCCESS, w.user_data);
      }
      return;
    }
    for (const auto& w : waiters) {
      w.callback(nullptr, final_error, w.user_data);
    }
  }

  static int GetCredentials(void* delegate_user_data,
                            aws_on_get_credentials_callback_fn* callback,
                            void* callback_user_data) {
    auto* self = static_cast<FirstResolvedState*>(delegate_user_data);
    AwsCredentialsProvider latched;
    AwsCredentialsProvider first_to_try;
    {
      absl::MutexLock lock(self->mutex_);
      if (self->mode_ == Mode::kLatched) {
        latched = self->latched_;
      } else {
        self->pending_.push_back({callback, callback_user_data});
        if (self->mode_ == Mode::kUnresolved) {
          self->mode_ = Mode::kResolving;
          self->current_index_ = 0;
          first_to_try = self->providers_[0];
        }
      }
    }
    if (latched) {
      if (aws_credentials_provider_get_credentials(
              latched.get(), callback, callback_user_data) != AWS_OP_SUCCESS) {
        callback(nullptr, SyncFailureErrorCode(), callback_user_data);
      }
      return AWS_OP_SUCCESS;
    }
    if (first_to_try) {
      QueryCurrentProvider(internal::IntrusivePtr<FirstResolvedState>(self),
                           std::move(first_to_try));
    }
    return AWS_OP_SUCCESS;
  }

  absl::Mutex mutex_;
  Mode mode_ ABSL_GUARDED_BY(mutex_) = Mode::kUnresolved;
  size_t current_index_ ABSL_GUARDED_BY(mutex_) = 0;
  std::vector<AwsCredentialsProvider> providers_ ABSL_GUARDED_BY(mutex_);
  AwsCredentialsProvider latched_ ABSL_GUARDED_BY(mutex_);
  std::vector<PendingQuery> pending_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace

AwsCredentialsProvider MakeCache(AwsCredentialsProvider provider,
                                 absl::Duration refresh_time) {
  if (!provider || refresh_time <= absl::ZeroDuration()) return provider;
  aws_allocator* allocator = GetAwsAllocator();

  aws_credentials_provider_cached_options options;
  AWS_ZERO_STRUCT(options);
  options.source = provider.get();
  if (refresh_time < absl::InfiniteDuration()) {
    options.refresh_time_in_milliseconds = static_cast<uint64_t>(
        std::max<int64_t>(1, absl::ToInt64Milliseconds(refresh_time)));
  }
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

AwsCredentialsProvider MakeFirstResolved(
    absl::Span<const AwsCredentialsProvider> providers) {
  return FirstResolvedState::Make(providers);
}

AwsCredentialsProvider MakeDefaultWithAnonymous(
    std::string_view profile_name_override) {
  auto default_provider = MakeDefault(profile_name_override);
  if (!default_provider) return nullptr;
  auto anonymous_provider = MakeAnonymous();
  if (!anonymous_provider) return nullptr;
  return MakeFirstResolved({default_provider, anonymous_provider});
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

  auto parsed = internal_uri::ParseGenericUri(endpoint);
  auto host_port = internal_uri::SplitHostPort(parsed.authority);
  if (!host_port) {
    return nullptr;
  }
  if (!parsed.authority.empty()) {
    if (!internal_uri::SplitHostPort(parsed.authority)) {
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
