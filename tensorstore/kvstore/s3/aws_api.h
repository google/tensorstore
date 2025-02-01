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

#ifndef TENSORSTORE_KVSTORE_S3_AWS_API_H_
#define TENSORSTORE_KVSTORE_S3_AWS_API_H_

#include <memory>

#include <aws/common/allocator.h>
#include <aws/io/channel_bootstrap.h>
#include <aws/io/host_resolver.h>
#include <aws/io/tls_channel_handler.h>

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Context class for using the AWS apis.
class AwsApiContext {
 public:
  ~AwsApiContext();

  aws_allocator *allocator;  // global
  aws_host_resolver *resolver;
  aws_client_bootstrap *client_bootstrap;
  aws_tls_ctx *tls_ctx;
};

std::shared_ptr<AwsApiContext> GetAwsApiContext();

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_AWS_API_H_
