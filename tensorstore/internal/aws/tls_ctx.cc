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

#include "tensorstore/internal/aws/tls_ctx.h"

#include <stddef.h>
#include <stdint.h>

#include <string_view>

#include <aws/common/allocator.h>
#include <aws/common/byte_buf.h>
#include <aws/common/zero.h>
#include <aws/io/tls_channel_handler.h>
#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {
namespace internal_aws {

AwsTlsCtxBuilder::AwsTlsCtxBuilder(aws_allocator *allocator) {
  AWS_ZERO_STRUCT(tls_context_options);
  aws_tls_ctx_options_init_default_client(&tls_context_options, allocator);
}

AwsTlsCtxBuilder::~AwsTlsCtxBuilder() {
  aws_tls_ctx_options_clean_up(&tls_context_options);
}

AwsTlsCtxBuilder &AwsTlsCtxBuilder::OverrideDefaultTrustStore(
    std::string_view pem_certificate) {
  aws_byte_cursor cursor;
  cursor.ptr =
      reinterpret_cast<uint8_t *>(const_cast<char *>(pem_certificate.data()));
  cursor.len = pem_certificate.size();
  aws_tls_ctx_options_override_default_trust_store(&tls_context_options,
                                                   &cursor);
  return *this;
}

AwsTlsCtx AwsTlsCtxBuilder::Build() && {
  return AwsTlsCtx(aws_tls_client_ctx_new(tls_context_options.allocator,
                                          &tls_context_options),
                   internal::adopt_object_ref);
}

// TODO input_stream from cord
// Add refcount.on_zero_fn

}  // namespace internal_aws
}  // namespace tensorstore
