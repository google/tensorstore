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

#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/kvstore/s3/credentials/ec2_credential_provider.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

Result<AwsCredentials>
EC2MetadataCredentialProvider::GetCredentials()
{
  return absl::NotFoundError("EC2 Metadata Server");
}


} // namespace internal_kvstore_s3
} // namespace tensorstore
