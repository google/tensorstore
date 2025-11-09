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

#include "tensorstore/internal/aws/aws_credentials.h"

#include <stdint.h>

#include <cassert>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include <aws/auth/credentials.h>
#include <aws/common/error.h>
#include "tensorstore/internal/aws/aws_api.h"
#include "tensorstore/internal/aws/string_view.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_aws {
namespace {

struct CallbackData {
  CallbackData(AwsCredentialsProvider provider, Promise<AwsCredentials> promise)
      : provider(std::move(provider)), promise(std::move(promise)) {}

  AwsCredentialsProvider provider;
  Promise<AwsCredentials> promise;
};

void OnGetCredentialsCallback(aws_credentials* credentials, int error_code,
                              void* user_data) {
  std::unique_ptr<CallbackData> data(
      reinterpret_cast<CallbackData*>(user_data));
  if (error_code == AWS_OP_SUCCESS && credentials != nullptr) {
    data->promise.SetResult(AwsCredentials(credentials));
    return;
  }
  // NOTE: Improve mapping from AWS error codes to absl::Status codes.
  if (error_code == AWS_OP_SUCCESS) {
    // No credentials returned.  See if the last error is useful.
    error_code = aws_last_error();
  }
  if (error_code != AWS_ERROR_SUCCESS) {
    data->promise.SetResult(absl::InternalError(
        absl::StrCat("Failed to get credentials from provider: ",
                     aws_error_debug_str(error_code))));
  } else {
    data->promise.SetResult(absl::InternalError(
        "Failed to get credentials from provider: no credentials returned"));
  }
}

}  // namespace

Future<AwsCredentials> GetAwsCredentials(aws_credentials_provider* provider) {
  if (!provider) return AwsCredentials(nullptr);

  auto p = PromiseFuturePair<AwsCredentials>::Make();
  auto state = std::make_unique<CallbackData>(AwsCredentialsProvider(provider),
                                              std::move(p.promise));

  auto error_code = aws_credentials_provider_get_credentials(
      provider, &OnGetCredentialsCallback, state.get());
  if (error_code == AWS_OP_SUCCESS) {
    state.release();
    return std::move(p.future);
  }
  return absl::InternalError(
      absl::StrCat("Failed to get credentials from provider: ",
                   aws_error_debug_str(error_code)));
}

std::string_view AwsCredentials::GetAccessKeyId() const {
  if (!get()) return {};
  return AwsByteCursorToStringView(aws_credentials_get_access_key_id(get()));
}

std::string_view AwsCredentials::GetSecretAccessKey() const {
  if (!get()) return {};
  return AwsByteCursorToStringView(
      aws_credentials_get_secret_access_key(get()));
}

std::string_view AwsCredentials::GetSessionToken() const {
  if (!get()) return {};
  return AwsByteCursorToStringView(aws_credentials_get_session_token(get()));
}

absl::Time AwsCredentials::GetExpiration() const {
  if (!get()) return absl::InfiniteFuture();
  auto seconds = aws_credentials_get_expiration_timepoint_seconds(get());
  return seconds == std::numeric_limits<uint64_t>::max()
             ? absl::InfiniteFuture()
             : absl::FromUnixSeconds(seconds);
}

bool AwsCredentials::IsAnonymous() const {
  if (!get()) return true;
  return aws_credentials_is_anonymous(get());
}

/* static */
AwsCredentials AwsCredentials::Make(std::string_view access_key_id,
                                    std::string_view secret_access_key,
                                    std::string_view session_token,
                                    absl::Time expiration) {
  uint64_t expiration_seconds = (expiration == absl::InfiniteFuture())
                                    ? std::numeric_limits<uint64_t>::max()
                                    : absl::ToUnixSeconds(expiration);

  return AwsCredentials(
      aws_credentials_new(
          GetAwsAllocator(), StringViewToAwsByteCursor(access_key_id),
          StringViewToAwsByteCursor(secret_access_key),
          StringViewToAwsByteCursor(session_token), expiration_seconds),
      internal::adopt_object_ref);
}

}  // namespace internal_aws
}  // namespace tensorstore
