// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/kvstore/http/http_resource.h"

#include <stddef.h>

#include <memory>
#include <optional>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/rate_limiter/admission_queue.h"
#include "tensorstore/util/result.h"

/// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep

ABSL_FLAG(std::optional<size_t>, tensorstore_http_request_concurrency,
          std::nullopt,
          "Maximum HTTP Request Concurrency. "
          "Overrides TENSORSTORE_HTTP_REQUEST_CONCURRENCY");

using ::tensorstore::internal::AdmissionQueue;
using ::tensorstore::internal::ContextResourceCreationContext;
using ::tensorstore::internal::GetFlagOrEnvValue;

namespace tensorstore {
namespace internal_http {
namespace {

const internal::ContextResourceRegistration<HttpRequestRetries>
    http_request_retries_registration;

const internal::ContextResourceRegistration<HttpRequestConcurrencyResource>
    http_request_concurrency_registration;

ABSL_CONST_INIT internal_log::VerboseFlag http_logging("http_kvstore");

constexpr size_t kDefaultRequestConcurrency = 32;

size_t GetEnvHttpRequestConcurrency() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_http_request_concurrency,
                           "TENSORSTORE_HTTP_REQUEST_CONCURRENCY")
      .value_or(kDefaultRequestConcurrency);
}

}  // namespace

HttpRequestConcurrencyResource::HttpRequestConcurrencyResource(
    size_t shared_limit)
    : shared_limit_(shared_limit) {}

HttpRequestConcurrencyResource::HttpRequestConcurrencyResource()
    : HttpRequestConcurrencyResource(GetEnvHttpRequestConcurrency()) {}

Result<HttpRequestConcurrencyResource::Resource>
HttpRequestConcurrencyResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  if (spec.limit) {
    Resource value;
    value.spec = spec;
    value.queue = std::make_shared<AdmissionQueue>(*spec.limit);
    return value;
  }

  absl::call_once(shared_once_, [&] {
    ABSL_LOG_IF(INFO, http_logging)
        << "Using default AdmissionQueue with limit " << shared_limit_;
    shared_resource_.queue = std::make_shared<AdmissionQueue>(shared_limit_);
  });
  return shared_resource_;
}

}  // namespace internal_http
}  // namespace tensorstore
