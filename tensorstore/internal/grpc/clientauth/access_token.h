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

#ifndef TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_H_
#define TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_H_

#include <iostream>
#include <string>
#include <string_view>
#include <tuple>

#include "absl/time/time.h"

namespace tensorstore {
namespace internal_grpc {

/// Represents an access token with a known expiration time, used by the
/// ImpersonateServiceAccount strategy and AccessTokenCache.
struct AccessToken {
  std::string token;
  absl::Time expiration = absl::InfinitePast();

  friend std::ostream& operator<<(std::ostream& os, const AccessToken& rhs) {
    // Tokens are truncated because they contain security secrets.
    return os << "token=<" << std::string_view(rhs.token).substr(0, 32)
              << ">, expiration=" << absl::FormatTime(rhs.expiration);
  }

  friend bool operator==(const AccessToken& lhs, const AccessToken& rhs) {
    return std::tie(lhs.token, lhs.expiration) ==
           std::tie(rhs.token, rhs.expiration);
  }

  friend bool operator!=(const AccessToken& lhs, const AccessToken& rhs) {
    return !(lhs == rhs);
  }
};

}  // namespace internal_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_GRPC_CLIENTAUTH_ACCESS_TOKEN_H_
