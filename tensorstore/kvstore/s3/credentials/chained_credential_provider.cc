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

#include <memory>
#include <string>
#include <vector>

#include "tensorstore/util/result.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "tensorstore/util/result.h"
#include "tensorstore/kvstore/s3/credentials/chained_credential_provider.h"

namespace tensorstore {
namespace internal_kvstore_s3 {


bool ChainedCredentialProvider::IsExpired()  {
    if(!LastProviderValid()) {
        return true;
    }

    return providers_[last_provider_]->IsExpired();
}

Result<absl::Time> ChainedCredentialProvider::ExpiresAt() {
    if(!LastProviderValid()) {
        return absl::UnimplementedError("ChainedCredentialProvider::ExpiresAt");
    }

    return providers_[last_provider_]->ExpiresAt();
}

Result<AwsCredentials> ChainedCredentialProvider::GetCredentials() {
    std::vector<std::string> errors;
    last_provider_ = -1;

    for(std::size_t i=0; i < providers_.size(); ++i) {
        auto credentials = providers_[i]->GetCredentials();
        if(credentials.ok()) {
            last_provider_ = i;
            return credentials;
        } else {
            errors.push_back(credentials.status().ToString());
        }
    }

    return absl::NotFoundError(
        absl::StrCat("No valid AwsCredentialProvider in chain:\n",
                     absl::StrJoin(errors, "\n")));
}


} // namespace internal_kvstore_s3
} // namespace tensorstore
