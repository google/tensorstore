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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_FIXED_TOKEN_AUTH_PROVIDER_H_
#define TENSORSTORE_INTERNAL_OAUTH2_FIXED_TOKEN_AUTH_PROVIDER_H_

#include <string>

#include "tensorstore/internal/oauth2/auth_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

class FixedTokenAuthProvider : public AuthProvider {
 public:
  ~FixedTokenAuthProvider() override = default;

  FixedTokenAuthProvider(std::string token);

  /// \brief Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  Result<BearerTokenWithExpiration> GetToken() override;

 private:
  std::string token_;
};

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_FIXED_TOKEN_AUTH_PROVIDER_H_
