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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_AUTH_PROVIDER_H_
#define TENSORSTORE_INTERNAL_OAUTH2_AUTH_PROVIDER_H_

#include <string>

#include "absl/time/time.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_oauth2 {

class AuthProvider {
 public:
  virtual ~AuthProvider();

  // Bundles an OAuth bearer token along with an expiration timestamp.
  struct BearerTokenWithExpiration {
    std::string token;
    absl::Time expiration;
  };

  /// \brief Returns the short-term authentication bearer token.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual Result<BearerTokenWithExpiration> GetToken() = 0;

  /// \brief Returns the header for the Token
  Result<std::string> GetAuthHeader();

  static constexpr absl::Duration kExpirationMargin = absl::Seconds(60);
};

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_AUTH_PROVIDER_H_
