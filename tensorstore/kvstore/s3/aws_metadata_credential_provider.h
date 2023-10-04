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

#ifndef TENSORSTORE_KVSTORE_S3_AWS_METADATA_CREDENTIAL_PROVIDER_H
#define TENSORSTORE_KVSTORE_S3_AWS_METADATA_CREDENTIAL_PROVIDER_H

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3/aws_credential_provider.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Provide S3 credentials from the EC2 Metadata server
class EC2MetadataCredentialProvider : public AwsCredentialProvider {
 public:
  EC2MetadataCredentialProvider(
      std::shared_ptr<internal_http::HttpTransport> transport)
      : transport_(std::move(transport)), timeout_(absl::InfinitePast()) {}

  Result<AwsCredentials> GetCredentials() override;

 private:
  std::shared_ptr<internal_http::HttpTransport> transport_;

  absl::Mutex mutex_;
  absl::Time timeout_ ABSL_GUARDED_BY(mutex_);
  AwsCredentials credentials_ ABSL_GUARDED_BY(mutex_);
};

// Returns whether the EC2 Metadata Server is available.
bool IsEC2MetadataServiceAvailable(internal_http::HttpTransport& transport);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_AWS_METADATA_CREDENTIAL_PROVIDER_H
