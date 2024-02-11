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

#include "tensorstore/kvstore/s3/s3_resource.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/base/call_once.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/kvstore/gcs_http/admission_queue.h"
#include "tensorstore/kvstore/gcs_http/rate_limiter.h"
#include "tensorstore/kvstore/gcs_http/scaling_rate_limiter.h"
#include "tensorstore/util/result.h"

/// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep

ABSL_FLAG(std::optional<size_t>, tensorstore_s3_request_concurrency,
          std::nullopt,
          "Maximum S3 Request Concurrency. Unbounded if not set. "
          "Overrides TENSORSTORE_S3_REQUEST_CONCURRENCY");

ABSL_FLAG(std::optional<absl::Duration>,
          tensorstore_s3_rate_limiter_doubling_time, std::nullopt,
          "S3 Rate Limiter Doubling Time. "
          "Overrides TENSORSTORE_S3_RATE_LIMITER_DOUBLING_TIME");

using ::tensorstore::internal::AnyContextResourceJsonBinder;
using ::tensorstore::internal::ContextResourceCreationContext;
using ::tensorstore::internal_kvstore_gcs_http::AdmissionQueue;
using ::tensorstore::internal_kvstore_gcs_http::NoRateLimiter;
using ::tensorstore::internal_kvstore_gcs_http::ScalingRateLimiter;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

const internal::ContextResourceRegistration<S3RequestRetries>
    s3_request_retries_registration;

const internal::ContextResourceRegistration<S3ConcurrencyResource>
    s3_concurrency_registration;

const internal::ContextResourceRegistration<S3RateLimiterResource>
    s3_rate_limiter_registration;

constexpr size_t kDefaultRequestConcurrency = 32;

size_t GetEnvS3RequestConcurrency() {
  if (auto var = absl::GetFlag(FLAGS_tensorstore_s3_request_concurrency); var) {
    return *var;
  }

  return internal::GetEnvValue<size_t>("TENSORSTORE_S3_REQUEST_CONCURRENCY")
      .value_or(kDefaultRequestConcurrency);
}

absl::Duration GetEnvS3RateLimiterDoublingTime() {
  if (auto var = absl::GetFlag(FLAGS_tensorstore_s3_rate_limiter_doubling_time);
      var) {
    return *var;
  }

  if (auto env = internal::GetEnv("TENSORSTORE_S3_RATE_LIMITER_DOUBLING_TIME");
      env) {
    absl::Duration doubling;
    std::string error;
    if (absl::ParseFlag(*env, &doubling, &error)) {
      return doubling;
    }
  }

  return absl::ZeroDuration();
}

}  // namespace

S3ConcurrencyResource::S3ConcurrencyResource(size_t shared_limit)
    : shared_limit_(shared_limit) {}

S3ConcurrencyResource::S3ConcurrencyResource()
    : S3ConcurrencyResource(GetEnvS3RequestConcurrency()) {}

Result<S3ConcurrencyResource::Resource> S3ConcurrencyResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  if (spec.limit) {
    Resource value;
    value.spec = spec;
    value.queue = std::make_shared<AdmissionQueue>(*spec.limit);
    return value;
  }

  absl::call_once(shared_once_, [&] {
    ABSL_LOG(INFO) << "Using default AdmissionQueue with limit "
                   << shared_limit_;
    shared_resource_.queue = std::make_shared<AdmissionQueue>(shared_limit_);
  });
  return shared_resource_;
}

Result<S3RateLimiterResource::Resource> S3RateLimiterResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  Resource value;
  value.spec = spec;
  if (spec.read_rate) {
    value.read_limiter = std::make_shared<ScalingRateLimiter>(
        *spec.read_rate, *spec.read_rate * 2,
        spec.doubling_time.value_or(GetEnvS3RateLimiterDoublingTime()));
  } else {
    value.read_limiter = std::make_shared<NoRateLimiter>();
  }
  if (spec.write_rate) {
    value.write_limiter = std::make_shared<ScalingRateLimiter>(
        *spec.write_rate, *spec.read_rate * 2,
        spec.doubling_time.value_or(GetEnvS3RateLimiterDoublingTime()));
  } else {
    value.write_limiter = std::make_shared<NoRateLimiter>();
  }
  return value;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
