// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/oauth2/fixed_token_auth_provider.h"

#include "absl/time/time.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

FixedTokenAuthProvider::FixedTokenAuthProvider(std::string token)
    : token_(token) {}

Result<AuthProvider::BearerTokenWithExpiration>
FixedTokenAuthProvider::GetToken() {
  return BearerTokenWithExpiration{token_, absl::InfiniteFuture()};
}

}  // namespace internal_oauth2
}  // namespace tensorstore
