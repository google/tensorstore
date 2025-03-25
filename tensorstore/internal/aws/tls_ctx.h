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

#ifndef TENSORSTORE_INTERNAL_AWS_TLS_CTX_H_
#define TENSORSTORE_INTERNAL_AWS_TLS_CTX_H_

#include <string_view>

#include <aws/common/allocator.h>
#include <aws/io/tls_channel_handler.h>
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {
namespace internal_aws {

struct AwsTlsCtxTraits {
  template <typename U>
  using pointer = U *;
  static void increment(aws_tls_ctx *p) noexcept { aws_tls_ctx_acquire(p); }
  static void decrement(aws_tls_ctx *p) noexcept { aws_tls_ctx_release(p); }
};

using AwsTlsCtx = internal::IntrusivePtr<aws_tls_ctx, AwsTlsCtxTraits>;

/// Builder for an AWS TLS context.
class AwsTlsCtxBuilder {
 public:
  AwsTlsCtxBuilder(aws_allocator *allocator);
  ~AwsTlsCtxBuilder();

  AwsTlsCtxBuilder &OverrideDefaultTrustStore(std::string_view pem_certificate);

  /// Builds an AWS TLS context.
  AwsTlsCtx Build() &&;

 private:
  aws_tls_ctx_options tls_context_options;
};

}  // namespace internal_aws
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_AWS_TLS_CTX_H_
