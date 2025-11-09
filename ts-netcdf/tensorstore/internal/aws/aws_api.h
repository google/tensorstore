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

#ifndef TENSORSTORE_INTERNAL_AWS_AWS_API_H_
#define TENSORSTORE_INTERNAL_AWS_AWS_API_H_

#include <aws/common/allocator.h>
#include <aws/io/channel_bootstrap.h>
#include <aws/io/tls_channel_handler.h>

namespace tensorstore {
namespace internal_aws {

/// Returns the global AWS allocator.
aws_allocator *GetAwsAllocator();

/// Returns the global AWS client bootstrap.
aws_client_bootstrap *GetAwsClientBootstrap();

/// Returns the global AWS TLS context.
aws_tls_ctx *GetAwsTlsCtx();

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_AWS_AWS_API_H_
